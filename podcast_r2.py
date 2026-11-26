"""Upload only the MP3s referenced by local Sabia chapter manifests."""

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    ConnectionError as BotoConnectionError,
    CredentialRetrievalError,
    HTTPClientError,
    NoCredentialsError,
    PartialCredentialsError,
)
from dotenv import load_dotenv

from podcast_manifest import sha256_file


MULTIPART_THRESHOLD = 64 * 1024 * 1024
PART_SIZE = 16 * 1024 * 1024
MAX_OBJECT_SIZE = 5 * 1024**4


class UploadError(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


@dataclass(frozen=True)
class AudioFile:
    key: str
    path: Path
    sha256: str
    size: int


@dataclass
class UploadSummary:
    manifest: Path
    uploaded: list[str] = field(default_factory=list)
    existing: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return bool(self.uploaded or self.existing) and not self.failed and not self.errors


def _safe_key(key: object) -> str:
    if not isinstance(key, str) or not key or len(key.encode("utf-8")) > 1024:
        raise UploadError("local", "r2_key debe ser una clave relativa no vacia.")
    for part in key.split("/"):
        if (
            part in ("", ".", "..")
            or part.endswith((" ", "."))
            or re.search(r'[\\:<>"|?*\x00-\x1f\x7f]', part)
            or re.fullmatch(r"(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part, re.I)
        ):
            raise UploadError("local", "r2_key contiene una ruta insegura.")
    return key


def _manifest_files(manifest: object, directory: Path) -> list[tuple[str, Path]]:
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise UploadError("local", "El manifest debe tener schema_version: 1.")
    audio = manifest.get("audio")
    if not isinstance(audio, dict):
        raise UploadError("local", "Falta el objeto audio.")
    key = _safe_key(audio.get("r2_key"))
    if not re.fullmatch(r"[^/]+/[a-z]{2}/(ch[0-9]{2,})/v[1-9][0-9]*/\1\.mp3", key):
        raise UploadError("local", "audio.r2_key no corresponde a una ruta versionada del generador.")
    if not isinstance(audio.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", audio["sha256"]):
        raise UploadError("local", "audio.sha256 debe ser un SHA-256 completo en minusculas.")
    duration = audio.get("duration_sec")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or duration <= 0
        or duration > sys.float_info.max
        or not math.isfinite(duration)
    ):
        raise UploadError("local", "audio.duration_sec debe ser finito y positivo.")
    prefix, filename = key.rsplit("/", 1)
    references = [(key, directory / filename)]
    quiz = manifest.get("quiz", [])
    if not isinstance(quiz, list):
        raise UploadError("local", "quiz debe ser un array o estar ausente.")
    for entry in quiz:
        if not isinstance(entry, dict):
            raise UploadError("local", "Cada segmento de quiz debe ser un objeto.")
        quiz_key = _safe_key(entry.get("r2_key"))
        tag = entry.get("tag")
        if not isinstance(tag, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", tag):
            raise UploadError("local", "Etiqueta de quiz no valida.")
        relative = f"quiz/{tag.lower()}.mp3"
        if quiz_key != f"{prefix}/{relative}":
            raise UploadError("local", "Clave de quiz incompatible con su etiqueta o version.")
        references.append((quiz_key, directory / "quiz" / f"{tag.lower()}.mp3"))
    return references


def _prepare(manifest_path: Path, snapshots: Path) -> list[AudioFile]:
    try:
        with manifest_path.open(encoding="utf-8") as source:
            manifest = json.load(source)
        directory = manifest_path.parent.resolve(strict=True)
        references = _manifest_files(manifest, directory)
        prepared: dict[str, AudioFile] = {}
        for key, path in references:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(directory) or not resolved.is_file():
                raise UploadError("local", "MP3 fuera del directorio del manifest o no regular.")
            if key in prepared:
                previous = prepared[key]
                if previous.sha256 != sha256_file(resolved) or previous.size != resolved.stat().st_size:
                    raise UploadError("local", "Clave duplicada con contenido incompatible.")
                continue
            # Disk snapshots keep preflight hashes and uploaded bytes identical,
            # even if a local generation is edited while the upload is running.
            snapshot = snapshots / str(len(prepared))
            shutil.copyfile(resolved, snapshot)
            size = snapshot.stat().st_size
            if not 0 < size <= MAX_OBJECT_SIZE:
                raise UploadError("local", "MP3 vacio o mayor que el limite de R2 (5 TiB).")
            prepared[key] = AudioFile(key, snapshot, sha256_file(snapshot), size)
        files = list(prepared.values())
        if files[0].sha256 != manifest["audio"]["sha256"]:
            raise UploadError("local", "El SHA-256 del audio principal no coincide con audio.sha256.")
        return files
    except (OSError, ValueError, UnicodeError) as error:
        raise UploadError("local", "No se pudo leer el manifest o alguno de sus MP3 locales.") from error


def _reject_redirects(response=None, **kwargs):
    if response is not None:
        http_response, _ = response
        content_type = http_response.headers.get("content-type", "").lower()
        if 300 <= http_response.status_code < 400 or "text/html" in content_type:
            raise UploadError(
                "conexion",
                "Redireccion o gateway HTML bloqueado. Comprueba proxy/Zscaler y acceso directo a R2; "
                "no se han seguido redirecciones ni mostrado datos de autenticacion.",
            )


def create_r2_client():
    try:
        load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
    except (OSError, UnicodeError) as error:
        raise UploadError("configuracion", "No se pudo leer el archivo .env local.") from error
    names = (
        "R2_ACCOUNT_ID", "R2_BUCKET_AUDIO",
        "R2_UPLOAD_ACCESS_KEY_ID", "R2_UPLOAD_SECRET_ACCESS_KEY",
    )
    values = {name: os.environ.get(name, "").strip() for name in names}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise UploadError("credenciales", f"Falta configuracion: {', '.join(missing)}.")
    if not re.fullmatch(r"[0-9a-fA-F]{32}", values["R2_ACCOUNT_ID"]):
        raise UploadError("configuracion", "R2_ACCOUNT_ID debe contener 32 caracteres hexadecimales.")
    bucket = values["R2_BUCKET_AUDIO"]
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise UploadError("configuracion", "R2_BUCKET_AUDIO debe ser un nombre de bucket, no una URL o ruta.")
    endpoint = f"https://{values['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name="auto",
        aws_access_key_id=values["R2_UPLOAD_ACCESS_KEY_ID"],
        aws_secret_access_key=values["R2_UPLOAD_SECRET_ACCESS_KEY"],
        aws_session_token="",
        verify=True,
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"mode": "standard", "total_max_attempts": 4},
            connect_timeout=10,
            read_timeout=120,
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            ignore_configured_endpoint_urls=True,
        ),
    )
    for operation in ("PutObject", "CompleteMultipartUpload"):
        shape = client.meta.service_model.operation_model(operation).input_shape
        if shape is None or "IfNoneMatch" not in shape.members:
            client.close()
            raise UploadError("configuracion", "Actualiza boto3: el SDK debe admitir IfNoneMatch en escrituras y multipart.")

    def check_destination(request, **kwargs):
        destination = urlsplit(request.url)
        if destination.scheme != "https" or destination.netloc != urlsplit(endpoint).netloc:
            raise UploadError("conexion", "Cambio de destino de autenticacion bloqueado.")

    client.meta.events.register_first("needs-retry.s3", _reject_redirects)
    client.meta.events.register_first("before-send.s3", check_destination)
    return client, bucket


def _remote_error(error: BotoCoreError | ClientError) -> UploadError:
    if isinstance(error, ClientError):
        code = error.response.get("Error", {}).get("Code", "")
        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        if code in ("InvalidAccessKeyId", "SignatureDoesNotMatch", "ExpiredToken", "InvalidToken", "AuthFailure") or status == 401:
            return UploadError("credenciales", "R2 rechazo las credenciales de subida.")
        if status == 403 or code == "AccessDenied":
            return UploadError("permisos", "R2 denego el acceso; comprueba los permisos del token sobre el bucket.")
        if status in (409, 412):
            return UploadError("conflicto", "La clave esta ocupada o bloqueada. Genera una nueva ruta de version.")
        if 300 <= status < 400:
            return UploadError("conexion", "Redireccion bloqueada; comprueba proxy/Zscaler y el endpoint R2.")
        if code == "NoSuchBucket":
            return UploadError("configuracion", "El bucket configurado no existe.")
        return UploadError("servicio", "R2 no completo la operacion tras los reintentos acotados.")
    if isinstance(error, (NoCredentialsError, PartialCredentialsError, CredentialRetrievalError)):
        return UploadError("credenciales", "No se pudieron cargar las credenciales R2.")
    if isinstance(error, (BotoConnectionError, HTTPClientError)):
        return UploadError("conexion", "Fallo de conexion/TLS con R2; comprueba red, certificados y proxy.")
    return UploadError("servicio", "Fallo del SDK R2; comprueba conectividad y compatibilidad de boto3.")


def _exists(client, bucket: str, audio: AudioFile) -> bool:
    try:
        head = client.head_object(Bucket=bucket, Key=audio.key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise
    if head.get("Metadata", {}).get("sha256") == audio.sha256 and head.get("ContentLength") == audio.size:
        return True
    raise UploadError(
        "conflicto",
        "Objeto existente diferente o sin SHA-256/tamano verificable. Genera una nueva ruta de version.",
    )


def _write(client, bucket: str, audio: AudioFile) -> None:
    common = {"Bucket": bucket, "Key": audio.key}
    metadata = {"ContentType": "audio/mpeg", "Metadata": {"sha256": audio.sha256}}
    with audio.path.open("rb") as source:
        if audio.size < MULTIPART_THRESHOLD:
            client.put_object(
                **common, **metadata, Body=source, ContentLength=audio.size, IfNoneMatch="*",
            )
            return
        upload_id = client.create_multipart_upload(**common, **metadata)["UploadId"]
        completed = False
        try:
            part_size = max(PART_SIZE, math.ceil(audio.size / 10000 / (1024 * 1024)) * 1024 * 1024)
            parts = []
            while chunk := source.read(part_size):
                number = len(parts) + 1
                part = client.upload_part(
                    **common, UploadId=upload_id, PartNumber=number, Body=chunk,
                )
                parts.append({"PartNumber": number, "ETag": part["ETag"]})
            client.complete_multipart_upload(
                **common, UploadId=upload_id, MultipartUpload={"Parts": parts}, IfNoneMatch="*",
            )
            completed = True
        finally:
            if not completed:
                try:
                    client.abort_multipart_upload(**common, UploadId=upload_id)
                except ClientError as error:
                    if error.response.get("Error", {}).get("Code") != "NoSuchUpload":
                        raise UploadError(
                            "servicio", "Fallo multipart y no se pudo abortar; revisa las subidas incompletas del bucket.",
                        ) from error
                except BotoCoreError as error:
                    raise UploadError(
                        "conexion", "Fallo multipart y no se pudo abortar; revisa las subidas incompletas del bucket.",
                    ) from error


def upload_manifest(manifest_path: Path, client, bucket: str) -> UploadSummary:
    """Validate a whole manifest before any network access; never upload its JSON."""
    manifest_path = Path(manifest_path)
    summary = UploadSummary(manifest_path)
    try:
        with tempfile.TemporaryDirectory(prefix="chatterbox-r2-") as temporary:
            files = _prepare(manifest_path, Path(temporary))
            for audio in files:
                try:
                    if _exists(client, bucket, audio):
                        summary.existing.append(audio.key)
                        continue
                    try:
                        _write(client, bucket, audio)
                    except ClientError as error:
                        status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
                        if status not in (409, 412, 404) or not _exists(client, bucket, audio):
                            raise
                        summary.existing.append(audio.key)
                    else:
                        summary.uploaded.append(audio.key)
                except UploadError as error:
                    summary.failed[audio.key] = f"{error.category}: {error}"
                except (BotoCoreError, ClientError) as error:
                    safe = _remote_error(error)
                    summary.failed[audio.key] = f"{safe.category}: {safe}"
                except OSError:
                    summary.failed[audio.key] = "local: No se pudo leer la copia temporal del MP3."
    except UploadError as error:
        summary.errors.append(f"{error.category}: {error}")
    except OSError:
        summary.errors.append("local: No se pudo crear o limpiar el almacenamiento temporal.")
    return summary


def format_summary(result: UploadSummary) -> str:
    lines = [
        f"{result.manifest.name}: subidos={len(result.uploaded)}, ya existentes={len(result.existing)}, "
        f"fallidos={len(result.failed)}, errores_manifest={len(result.errors)}, "
        f"listo para importar en Sabia={'SI' if result.ready else 'NO'}",
    ]
    lines += [f"  {key}: {message}" for key, message in result.failed.items()]
    lines += [f"  Validacion/manifest fallido: {message}" for message in result.errors]
    return "\n".join(lines)


def discover_manifests(publish_dir: Path) -> dict[str, Path]:
    """Map 'podcast/lang/chNN/vN/chNN.json' labels to published manifests."""
    root = Path(publish_dir).resolve()
    found = {}
    for path in sorted(root.glob("*/*/ch*/v*/ch*.json")):
        if path.is_file() and re.fullmatch(r"ch\d+\.json", path.name):
            found[path.relative_to(root).as_posix()] = path
    return found


def upload_selected(selected: list[str], publish_dir: Path) -> str:
    """Upload only manifests discovered under publish_dir (used by the Gradio button)."""
    if not selected:
        return "Selecciona al menos un capitulo."
    available = discover_manifests(publish_dir)
    if any(label not in available for label in selected):
        return "Seleccion no valida; actualiza la lista de capitulos."
    try:
        client, bucket = create_r2_client()
    except UploadError as error:
        return f"{error.category}: {error}"
    except (BotoCoreError, ClientError) as error:
        safe = _remote_error(error)
        return f"{safe.category}: {safe}"
    try:
        summaries = [upload_manifest(available[label], client, bucket) for label in selected]
    finally:
        client.close()
    return "\n\n".join(
        f"{label}\n{format_summary(summary)}" for label, summary in zip(selected, summaries)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+", type=Path, help="Manifests locales chNN.json (no se suben).")
    args = parser.parse_args(argv)
    try:
        client, bucket = create_r2_client()
    except UploadError as error:
        print(f"{error.category}: {error}", file=sys.stderr)
        return 1
    except (BotoCoreError, ClientError) as error:
        safe = _remote_error(error)
        print(f"{safe.category}: {safe}", file=sys.stderr)
        return 1
    success = True
    try:
        for manifest in args.manifests:
            result = upload_manifest(manifest, client, bucket)
            print(format_summary(result))
            success = success and result.ready
    finally:
        client.close()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
