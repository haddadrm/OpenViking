from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from openviking.parse.understanding_api import UnderstandingAPI, UnderstandingAPIError
from openviking_cli.exceptions import InvalidArgumentError


@pytest.mark.asyncio
async def test_parse_uses_downloaded_file_and_resolved_extension(monkeypatch, tmp_path):
    downloaded = tmp_path / "download"
    downloaded.write_bytes(b"%PDF-1.7")
    zip_path = tmp_path / "result.zip"
    zip_path.write_bytes(b"zip")
    uploaded: list[Path] = []

    api = UnderstandingAPI.__new__(UnderstandingAPI)
    api._video_exts = {"mp4"}
    api._audio_exts = {"mp3"}
    api._image_exts = {"png"}

    async def create_file(*, local_path):
        uploaded.append(local_path)
        return {"id": "file-1"}

    async def create_response_for_file(*, file_id):
        assert file_id == "file-1"
        return {"id": "response-1"}

    async def poll_response(*, response_id):
        assert response_id == "response-1"
        return {"status": "completed"}

    monkeypatch.setattr(api, "_create_file", create_file)
    monkeypatch.setattr(api, "_create_response_for_file", create_response_for_file)
    monkeypatch.setattr(api, "_poll_response", poll_response)
    monkeypatch.setattr(api, "_extract_zip_url", lambda _: "https://example.com/result.zip")
    monkeypatch.setattr(api, "_download_zip", lambda _: _return(zip_path))
    monkeypatch.setattr(
        api,
        "_unpack_zip_to_temp_dir",
        lambda **_: _return("viking://temp/result"),
    )

    result = await api.parse(
        downloaded,
        original_source="https://example.com/download?id=123",
        resource_name="report",
        resolved_extension=".pdf",
    )

    assert uploaded == [downloaded]
    assert result.source_path == "https://example.com/download?id=123"
    assert result.source_format == "pdf"
    assert result.root.title == "report"


@pytest.mark.asyncio
async def test_submit_file_validates_input_and_returns_response_id(tmp_path):
    empty_source = tmp_path / "empty.pdf"
    empty_source.touch()
    api = UnderstandingAPI.__new__(UnderstandingAPI)

    with pytest.raises(
        InvalidArgumentError,
        match="Understanding parser does not support empty files",
    ) as exc_info:
        await api.submit_file(empty_source)

    assert exc_info.value.code == "INVALID_ARGUMENT"

    source = tmp_path / "download.pdf"
    source.write_bytes(b"%PDF-1.7")
    api._create_file = AsyncMock(return_value={"id": "file-1"})
    api._create_response_for_file = AsyncMock(return_value={"id": "response-1"})

    response_id = await api.submit_file(source)

    assert response_id == "response-1"
    api._create_file.assert_awaited_once_with(local_path=source)
    api._create_response_for_file.assert_awaited_once_with(file_id="file-1")


@pytest.mark.asyncio
async def test_file_above_simple_limit_requires_resumable_upload(tmp_path):
    source = tmp_path / "large.pdf"
    source.write_bytes(b"123456789")
    api = UnderstandingAPI.__new__(UnderstandingAPI)
    api._upload_simple_max_bytes = 8
    api._enable_resumable_upload = False

    with pytest.raises(ValueError, match="size=9, upload_simple_max_bytes=8"):
        await api._create_file(local_path=source)


@pytest.mark.asyncio
async def test_file_above_simple_limit_uses_multipart_when_enabled(tmp_path):
    source = tmp_path / "large.pdf"
    source.write_bytes(b"123456789")
    api = UnderstandingAPI.__new__(UnderstandingAPI)
    api._upload_simple_max_bytes = 8
    api._enable_resumable_upload = True
    api._multipart_create_file = AsyncMock(return_value={"id": "file-1"})

    result = await api._create_file(local_path=source)

    assert result == {"id": "file-1"}
    api._multipart_create_file.assert_awaited_once_with(source)


@pytest.mark.asyncio
async def test_parse_failure_preserves_observed_remote_ids(monkeypatch, tmp_path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.7")
    api = UnderstandingAPI.__new__(UnderstandingAPI)
    api._video_exts = {"mp4"}
    api._audio_exts = {"mp3"}
    api._image_exts = {"png"}

    monkeypatch.setattr(api, "_create_file", AsyncMock(return_value={"id": "file-1"}))
    monkeypatch.setattr(
        api,
        "_create_response_for_file",
        AsyncMock(return_value={"id": "response-1"}),
    )
    monkeypatch.setattr(
        api,
        "_poll_response",
        AsyncMock(side_effect=RuntimeError("remote parse failed")),
    )

    with pytest.raises(UnderstandingAPIError, match="remote parse failed") as exc_info:
        await api.parse(source, source_name="original.pdf")

    assert exc_info.value.meta == {
        "doc_name": "original",
        "doc_type": "pdf",
        "source_name": "original.pdf",
        "file_name": "report.pdf",
        "file_id": "file-1",
        "response_id": "response-1",
    }
    assert "response-1" in str(exc_info.value)


async def _return(value):
    return value
