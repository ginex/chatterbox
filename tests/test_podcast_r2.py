import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from botocore.awsrequest import AWSResponse
from botocore.exceptions import ClientError, EndpointConnectionError, ReadTimeoutError, SSLError

import podcast_r2
from podcast_r2 import UploadError, create_r2_client, main, upload_manifest


def s3_error(code, status):
    return ClientError(
        {"Error": {"Code": code, "Message": "sensitive server message"},
         "ResponseMetadata": {"HTTPStatusCode": status}},
        "TestOperation",
    )


class MemoryR2:
    """Offline store enforcing the same create-only contract as R2."""

    def __init__(self):
        self.objects = {}
        self.calls = []
        self.before_write = None
        self.failures = {}
        self.multipart = {}

    def head_object(self, **args):
        self.calls.append(("head", args))
        if args["Key"] not in self.objects:
            raise s3_error("404", 404)
        obj = self.objects[args["Key"]]
        return {"ContentLength": len(obj["data"]), "Metadata": obj["metadata"], "ETag": "not-a-sha256"}

    def _commit(self, key, data, metadata, condition):
        if condition != "*":
            raise AssertionError("Missing create-only condition")
        if self.before_write:
            callback, self.before_write = self.before_write, None
            callback(key)
        if key in self.failures:
            raise self.failures.pop(key)
        if key in self.objects:
            raise s3_error("PreconditionFailed", 412)
        self.objects[key] = {"data": data, "metadata": metadata}

    def put_object(self, **args):
        self.calls.append(("put", {key: value for key, value in args.items() if key != "Body"}))
        if not hasattr(args["Body"], "read"):
            raise AssertionError("PutObject must stream from disk")
        self._commit(args["Key"], args["Body"].read(), args["Metadata"], args.get("IfNoneMatch"))

    def create_multipart_upload(self, **args):
        self.calls.append(("create", args))
        self.multipart["test-upload"] = {"args": args, "parts": []}
        return {"UploadId": "test-upload"}

    def upload_part(self, **args):
        self.calls.append(("part", {"size": len(args["Body"])}))
        self.multipart[args["UploadId"]]["parts"].append(args["Body"])
        return {"ETag": f"part-{args['PartNumber']}"}

    def complete_multipart_upload(self, **args):
        self.calls.append(("complete", args))
        upload = self.multipart[args["UploadId"]]
        self._commit(
            args["Key"], b"".join(upload["parts"]), upload["args"]["Metadata"], args.get("IfNoneMatch"),
        )
        del self.multipart[args["UploadId"]]

    def abort_multipart_upload(self, **args):
        self.calls.append(("abort", args))
        self.multipart.pop(args["UploadId"], None)

    def close(self):
        pass


class R2ManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        # The local directory deliberately does not contain the remote prefix.
        self.directory = Path(temporary.name) / "moved-chapter"
        self.directory.mkdir()
        self.main_audio = self.directory / "ch02.mp3"
        self.main_audio.write_bytes(b"main audio fixture")
        self.path = self.directory / "ch02.json"
        self.prefix = "clode-en-route/fr/ch02/v1"
        self.key = f"{self.prefix}/ch02.mp3"
        self.manifest = {
            "schema_version": 1,
            "audio": {
                "r2_key": self.key,
                "sha256": hashlib.sha256(self.main_audio.read_bytes()).hexdigest(),
                "duration_sec": 226.368,
            },
            "quiz": [],
        }
        self.store = MemoryR2()

    def save(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def add_quiz(self):
        tags = (
            "PREGUNTAS", "PREGUNTA_1", "OPCION_CORRECTA_1", "RESPUESTA_1", "CORRECTO",
            "INCORRECTO", "ESPERA_1", "ESPERA_2", "FIN_PREGUNTAS",
        )
        (self.directory / "quiz").mkdir(exist_ok=True)
        for tag in tags:
            relative = f"quiz/{tag.lower()}.mp3"
            (self.directory / relative).write_bytes(tag.encode("ascii"))
            self.manifest["quiz"].append({"tag": tag, "r2_key": f"{self.prefix}/{relative}"})

    def upload(self):
        self.save()
        return upload_manifest(self.path, self.store, "private-audio")

    def test_exact_referenced_objects_including_correct_option_and_no_other_files(self):
        self.add_quiz()
        (self.directory / "unreferenced.mp3").write_bytes(b"never upload")
        original = copy.deepcopy(self.manifest)
        result = self.upload()
        expected = {self.key} | {entry["r2_key"] for entry in self.manifest["quiz"]}
        self.assertTrue(result.ready)
        self.assertEqual(set(result.uploaded), expected)
        self.assertEqual(set(self.store.objects), expected)
        self.assertEqual(json.loads(self.path.read_text()), original)
        for key, obj in self.store.objects.items():
            self.assertEqual(obj["metadata"], {"sha256": hashlib.sha256(obj["data"]).hexdigest()})
        for operation, args in self.store.calls:
            if operation == "put":
                self.assertEqual(args["ContentType"], "audio/mpeg")
                self.assertNotIn("ACL", args)
                self.assertEqual(args["Bucket"], "private-audio")
        again = upload_manifest(self.path, self.store, "private-audio")
        self.assertTrue(again.ready)
        self.assertEqual(set(again.existing), expected)
        self.assertEqual(again.uploaded, [])

    def test_quiz_absent_empty_and_identical_duplicate_references(self):
        for quiz in (None, []):
            with self.subTest(quiz=quiz):
                if quiz is None:
                    self.manifest.pop("quiz", None)
                else:
                    self.manifest["quiz"] = quiz
                self.assertTrue(self.upload().ready)
        self.manifest["quiz"] = []
        self.add_quiz()
        self.manifest["quiz"].append(copy.deepcopy(self.manifest["quiz"][0]))
        result = self.upload()
        self.assertTrue(result.ready)
        self.assertEqual(len(result.uploaded), 9)

    def test_all_local_validation_finishes_before_any_network_call(self):
        self.add_quiz()
        original = copy.deepcopy(self.manifest)
        changes = (
            lambda m: m.update(schema_version=True),
            lambda m: m["audio"].update(sha256="0" * 64),
            lambda m: m["audio"].update(sha256="short"),
            lambda m: m["audio"].update(duration_sec=float("nan")),
            lambda m: m["audio"].update(duration_sec=10**400),
            lambda m: m.update(quiz=None),
            lambda m: m["quiz"][-1].update(r2_key=f"{self.prefix}/quiz/../escape.mp3"),
            lambda m: m["quiz"][-1].update(r2_key=m["quiz"][0]["r2_key"]),
            lambda m: m["quiz"][-1].update(r2_key="other/fr/ch02/v1/quiz/fin_capitulo.mp3"),
            lambda m: m["audio"].update(r2_key="C:\\private\\ch02.mp3"),
            lambda m: m["audio"].update(r2_key="/absolute/ch02.mp3"),
            lambda m: m["audio"].update(r2_key="demo/fr/ch02/v1/../../ch02.mp3"),
            lambda m: m["audio"].update(r2_key="CON/fr/ch02/v1/ch02.mp3"),
            lambda m: m["audio"].update(r2_key="demo./fr/ch02/v1/ch02.mp3"),
            lambda m: m["audio"].update(r2_key="demo/fr/ch02/v1/ch02.mp3:stream"),
        )
        for change in changes:
            self.manifest = copy.deepcopy(original)
            change(self.manifest)
            with self.subTest(manifest=self.manifest):
                result = self.upload()
                self.assertFalse(result.ready)
                self.assertTrue(result.errors)
                self.assertEqual(self.store.calls, [])
        self.manifest = original
        (self.directory / "quiz" / "fin_capitulo.mp3").unlink()
        result = self.upload()
        self.assertFalse(result.ready)
        self.assertEqual(self.store.calls, [])

    def test_missing_empty_and_malformed_local_files(self):
        self.path.write_text("{", encoding="utf-8")
        result = upload_manifest(self.path, self.store, "private-audio")
        self.assertTrue(result.errors)
        self.assertFalse(result.ready)
        self.main_audio.write_bytes(b"")
        self.assertTrue(self.upload().errors)
        self.main_audio.unlink()
        self.assertTrue(self.upload().errors)
        self.path.unlink()
        self.assertTrue(upload_manifest(self.path, self.store, "private-audio").errors)
        self.assertEqual(self.store.calls, [])

    def test_symlink_outside_chapter_is_rejected(self):
        outside = self.directory.parent / "outside.mp3"
        outside.write_bytes(self.main_audio.read_bytes())
        self.main_audio.unlink()
        try:
            self.main_audio.symlink_to(outside)
        except OSError:
            self.skipTest("Creating symlinks requires Windows developer mode or elevated privileges")
        self.assertTrue(self.upload().errors)
        self.assertEqual(self.store.calls, [])

    def test_existing_unverifiable_or_different_object_is_never_overwritten(self):
        data = self.main_audio.read_bytes()
        digest = self.manifest["audio"]["sha256"]
        for old in (
            {"data": data, "metadata": {}},
            {"data": data, "metadata": {"sha256": "0" * 64}},
            {"data": b"short", "metadata": {"sha256": digest}},
        ):
            with self.subTest(old=old):
                self.store.objects[self.key] = copy.deepcopy(old)
                result = self.upload()
                self.assertFalse(result.ready)
                self.assertIn("conflicto", result.failed[self.key])
                self.assertIn("nueva ruta de version", result.failed[self.key])
                self.assertEqual(self.store.objects[self.key], old)
        self.assertFalse(any(operation == "put" for operation, _ in self.store.calls))

    def test_concurrent_writer_same_content_is_skipped_different_content_conflicts(self):
        for same in (True, False):
            self.store = MemoryR2()
            competing = {
                "data": self.main_audio.read_bytes() if same else b"competing data",
                "metadata": {"sha256": self.manifest["audio"]["sha256"] if same else "0" * 64},
            }
            self.store.before_write = lambda key: self.store.objects.update({key: competing})
            result = self.upload()
            self.assertEqual(result.ready, same)
            self.assertEqual(result.existing, [self.key] if same else [])
            self.assertEqual(self.store.objects[self.key], competing)

    def test_partial_failure_is_reported_and_next_run_resumes(self):
        self.add_quiz()
        failed_key = self.manifest["quiz"][2]["r2_key"]
        self.store.failures[failed_key] = s3_error("AccessDenied", 403)
        first = self.upload()
        self.assertFalse(first.ready)
        self.assertIn("permisos", first.failed[failed_key])
        self.assertNotIn("sensitive", first.failed[failed_key])
        self.assertEqual(len(first.uploaded), 9)
        second = upload_manifest(self.path, self.store, "private-audio")
        self.assertTrue(second.ready)
        self.assertEqual(second.uploaded, [failed_key])
        self.assertEqual(len(second.existing), 9)

    def test_source_changes_after_preflight_do_not_change_uploaded_bytes(self):
        original = self.main_audio.read_bytes()
        self.store.before_write = lambda key: self.main_audio.write_bytes(b"edited")
        self.assertTrue(self.upload().ready)
        self.assertEqual(self.store.objects[self.key]["data"], original)

    def test_multipart_is_bounded_conditional_and_idempotent_under_races(self):
        data = b"a" * (2 * 1024 * 1024 + 7)
        self.main_audio.write_bytes(data)
        self.manifest["audio"]["sha256"] = hashlib.sha256(data).hexdigest()
        with patch.object(podcast_r2, "MULTIPART_THRESHOLD", 1), patch.object(podcast_r2, "PART_SIZE", 1024 * 1024):
            result = self.upload()
        self.assertTrue(result.ready)
        self.assertEqual(self.store.objects[self.key]["data"], data)
        sizes = [args["size"] for op, args in self.store.calls if op == "part"]
        self.assertEqual(sizes, [1024 * 1024, 1024 * 1024, 7])
        completion = next(args for op, args in self.store.calls if op == "complete")
        self.assertEqual(completion["IfNoneMatch"], "*")
        self.assertEqual(len(completion["MultipartUpload"]["Parts"]), 3)
        for same in (True, False):
            self.store = MemoryR2()
            self.store.before_write = lambda key: self.store.objects.update({
                key: {"data": data if same else b"other", "metadata": {"sha256": hashlib.sha256(data).hexdigest()}},
            })
            with patch.object(podcast_r2, "MULTIPART_THRESHOLD", 1):
                result = self.upload()
            self.assertEqual(result.ready, same)
            self.assertEqual(self.store.multipart, {})
            self.assertTrue(any(op == "abort" for op, _ in self.store.calls))

    def test_multipart_failure_aborts_parts_and_cleanup_failure_is_explicit(self):
        for abort_fails in (False, True):
            self.store = MemoryR2()
            self.store.upload_part = Mock(side_effect=s3_error("AccessDenied", 403))
            if abort_fails:
                self.store.abort_multipart_upload = Mock(side_effect=EndpointConnectionError(endpoint_url="secret"))
            with patch.object(podcast_r2, "MULTIPART_THRESHOLD", 1):
                result = self.upload()
            self.assertFalse(result.ready)
            self.assertEqual(self.store.objects, {})
            self.assertIn("no se pudo abortar" if abort_fails else "permisos", result.failed[self.key])
            self.assertNotIn("secret", result.failed[self.key])
            if not abort_fails:
                self.assertEqual(self.store.multipart, {})

    def test_error_categories_and_safe_diagnostics(self):
        errors = (
            (s3_error("SignatureDoesNotMatch", 403), "credenciales"),
            (s3_error("AccessDenied", 403), "permisos"),
            (s3_error("NoSuchBucket", 404), "configuracion"),
            (EndpointConnectionError(endpoint_url="https://secret"), "conexion"),
            (SSLError(endpoint_url="https://secret", error="secret certificate"), "conexion"),
            (ReadTimeoutError(endpoint_url="https://secret"), "conexion"),
            (s3_error("SlowDown", 503), "servicio"),
        )
        self.save()
        for error, category in errors:
            self.store.head_object = Mock(side_effect=error)
            result = upload_manifest(self.path, self.store, "private-audio")
            self.assertFalse(result.ready)
            self.assertTrue(result.failed[self.key].startswith(f"{category}:"))
            self.assertNotIn("secret", result.failed[self.key])
            self.assertNotIn("sensitive", result.failed[self.key])

    def test_cli_multiple_manifests_failure_exit_and_resume_summary(self):
        self.save()
        with patch.object(podcast_r2, "create_r2_client", return_value=(self.store, "private-audio")):
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                code = main([str(self.path), str(self.directory / "missing.json")])
            self.assertEqual(code, 1)
            self.assertIn("subidos=1, ya existentes=0, fallidos=0", output.getvalue())
            self.assertIn("listo para importar en Sabia=NO", output.getvalue())
            with patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(main([str(self.path)]), 0)
            self.assertIn("ya existentes=1", output.getvalue())
            self.assertIn("Sabia=SI", output.getvalue())


ENVIRONMENT = {
    "R2_ACCOUNT_ID": "a" * 32,
    "R2_BUCKET_AUDIO": "test-private",
    "R2_UPLOAD_ACCESS_KEY_ID": "test-only-id",
    "R2_UPLOAD_SECRET_ACCESS_KEY": "test-only-secret",
}


class RawResponse:
    def __init__(self, body):
        self.body = body

    def stream(self, **kwargs):
        yield self.body


class GradioSelectionTests(R2ManifestTests):
    def test_discovers_only_published_manifests_and_uploads_selection(self):
        publish = self.directory.parent / "publish"
        chapter = publish / "clode-en-route" / "fr" / "ch02" / "v1"
        chapter.parent.mkdir(parents=True)
        self.directory.rename(chapter)
        (chapter / "chapter.json").write_text("{}", encoding="utf-8")
        self.directory = chapter
        self.path = chapter / "ch02.json"
        self.save()
        label = "clode-en-route/fr/ch02/v1/ch02.json"
        self.assertEqual(list(podcast_r2.discover_manifests(publish)), [label])
        with patch.object(podcast_r2, "create_r2_client", return_value=(self.store, "private-audio")):
            report = podcast_r2.upload_selected([label], publish)
            self.assertIn("subidos=1", report)
            self.assertIn("Sabia=SI", report)
            self.assertIn("no valida", podcast_r2.upload_selected(["../outside.json"], publish))
        self.assertEqual(set(self.store.objects), {self.key})
        self.assertIn("al menos", podcast_r2.upload_selected([], publish))


for _inherited in [name for name in dir(R2ManifestTests) if name.startswith("test_")]:
    setattr(GradioSelectionTests, _inherited, None)  # reuse setUp/helpers only


class R2TransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        source = patch.object(podcast_r2, "__file__", str(self.directory / "podcast_r2.py"))
        source.start()
        self.addCleanup(source.stop)

    def client(self):
        with patch.dict("os.environ", ENVIRONMENT, clear=True):
            client, bucket = create_r2_client()
        self.addCleanup(client.close)
        return client

    def response(self, status, headers=None, body=b""):
        return AWSResponse(
            f"https://{'a' * 32}.r2.cloudflarestorage.com/test-private/file.mp3",
            status, headers or {}, RawResponse(body),
        )

    def test_dotenv_loads_local_credentials_without_overriding_environment(self):
        (self.directory / ".env").write_text(
            "\n".join(f"{key}={value}" for key, value in ENVIRONMENT.items()),
            encoding="utf-8",
        )
        for environment in ({}, {"R2_BUCKET_AUDIO": "shell-bucket"}):
            with self.subTest(environment=environment):
                with patch.dict("os.environ", environment, clear=True):
                    client, bucket = create_r2_client()
                    try:
                        self.assertEqual(bucket, environment.get("R2_BUCKET_AUDIO", "test-private"))
                        self.assertEqual(client.meta.region_name, "auto")
                        self.assertEqual(
                            client.meta.endpoint_url,
                            f"https://{'a' * 32}.r2.cloudflarestorage.com",
                        )
                    finally:
                        client.close()

    def test_unreadable_dotenv_reports_safe_configuration_error(self):
        (self.directory / ".env").write_bytes(b"R2_UPLOAD_SECRET_ACCESS_KEY=\xff")
        with patch.dict("os.environ", {}, clear=True):
            with patch.object(podcast_r2.boto3, "client") as factory:
                with self.assertRaisesRegex(UploadError, "No se pudo leer") as raised:
                    create_r2_client()
        self.assertEqual(raised.exception.category, "configuracion")
        factory.assert_not_called()

    def test_missing_or_invalid_config_does_not_create_client_or_expose_values(self):
        for override in (
            {"R2_UPLOAD_SECRET_ACCESS_KEY": ""},
            {"R2_ACCOUNT_ID": "https://secret.example"},
            {"R2_BUCKET_AUDIO": "test-private/prefix"},
        ):
            with patch.dict("os.environ", {**ENVIRONMENT, **override}, clear=True):
                with patch.object(podcast_r2.boto3, "client") as factory:
                    with self.assertRaises(UploadError) as raised:
                        create_r2_client()
            self.assertNotIn("test-only-secret", str(raised.exception))
            self.assertNotIn("secret.example", str(raised.exception))
            factory.assert_not_called()

    def test_sdk_serializes_create_only_headers_and_no_acl_with_tls(self):
        client = self.client()
        response = self.response(200)
        with patch.object(client._endpoint.http_session, "send", return_value=response) as send:
            client.put_object(
                Bucket="test-private", Key="exact/version/key.mp3", Body=io.BytesIO(b"audio"),
                ContentLength=5, ContentType="audio/mpeg", Metadata={"sha256": "b" * 64},
                IfNoneMatch="*",
            )
        request = send.call_args.args[0]
        self.assertEqual(request.headers["If-None-Match"], b"*")
        self.assertEqual(request.headers["Content-Type"], b"audio/mpeg")
        self.assertEqual(request.headers["x-amz-meta-sha256"], b"b" * 64)
        self.assertNotIn("x-amz-acl", request.headers)
        self.assertEqual(request.url, f"https://{'a' * 32}.r2.cloudflarestorage.com/test-private/exact/version/key.mp3")
        self.assertTrue(client._endpoint.http_session._verify)
        completed = self.response(200, body=b"<CompleteMultipartUploadResult><ETag>opaque</ETag></CompleteMultipartUploadResult>")
        with patch.object(client._endpoint.http_session, "send", return_value=completed) as send:
            client.complete_multipart_upload(
                Bucket="test-private", Key="file.mp3", UploadId="test-upload",
                MultipartUpload={"Parts": [{"PartNumber": 1, "ETag": "opaque"}]}, IfNoneMatch="*",
            )
        self.assertEqual(send.call_args.args[0].headers["If-None-Match"], b"*")

    def test_sdk_without_conditional_support_fails_closed(self):
        client = Mock()
        client.meta.service_model.operation_model.return_value.input_shape.members = {}
        with patch.dict("os.environ", ENVIRONMENT, clear=True):
            with patch.object(podcast_r2.boto3, "client", return_value=client):
                with self.assertRaisesRegex(UploadError, "Actualiza boto3"):
                    create_r2_client()
        client.close.assert_called_once()
        client.put_object.assert_not_called()
        client.create_multipart_upload.assert_not_called()

    def test_redirects_and_html_gateways_are_blocked_before_sdk_redirect_or_retry(self):
        for status, headers, body in (
            (302, {"location": "https://login.example/?token=secret"}, b""),
            (301, {"location": "https://login.example/?token=secret", "x-amz-bucket-region": "elsewhere"}, b""),
            (200, {"content-type": "text/html"}, b"<html>Zscaler secret login</html>"),
        ):
            client = self.client()
            with patch.object(client._endpoint.http_session, "send", return_value=self.response(status, headers, body)) as send:
                with self.assertRaises(UploadError) as raised:
                    client.head_object(Bucket="test-private", Key="file.mp3")
            self.assertEqual(send.call_count, 1)
            self.assertIn("Zscaler", str(raised.exception))
            self.assertNotIn("token=secret", str(raised.exception))
            self.assertNotIn("login.example", str(raised.exception))

    def test_unexpected_destination_never_sends_authenticated_request(self):
        client = self.client()

        def change_destination(request, **kwargs):
            request.url = "https://login.example/secret"

        client.meta.events.register("before-sign.s3", change_destination)
        with patch.object(client._endpoint.http_session, "send") as send:
            with self.assertRaisesRegex(UploadError, "destino"):
                client.head_object(Bucket="test-private", Key="file.mp3")
        send.assert_not_called()

    def test_transient_retries_are_bounded_and_can_recover(self):
        for recover in (True, False):
            client = self.client()
            error = self.response(503, body=b"<Error><Code>SlowDown</Code></Error>")
            responses = [error, self.response(200)] if recover else [error] * 4
            with patch.object(client._endpoint.http_session, "send", side_effect=responses) as send:
                with patch("botocore.endpoint.time.sleep"):
                    if recover:
                        client.head_object(Bucket="test-private", Key="file.mp3")
                    else:
                        with self.assertRaises(ClientError):
                            client.head_object(Bucket="test-private", Key="file.mp3")
            self.assertEqual(send.call_count, 2 if recover else 4)


if __name__ == "__main__":
    unittest.main()
