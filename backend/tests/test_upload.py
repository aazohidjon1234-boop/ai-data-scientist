"""CSV upload: validation and happy path."""
from tests.conftest import upload_csv


def test_upload_valid_csv(client):
    content = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
    r = upload_csv(client, content, "valid.csv")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["rows"] == 3
    assert body["columns"] == 3
    assert body["column_names"] == ["a", "b", "c"]
    assert body["status"] == "uploaded"
    assert body["has_analysis"] is False


def test_uploaded_dataset_appears_in_list(client):
    upload_csv(client, "a,b\n1,2\n3,4\n", "listed.csv")
    r = client.get("/api/datasets")
    assert r.status_code == 200
    names = [d["original_filename"] for d in r.json()]
    assert "listed.csv" in names


def test_upload_empty_file_rejected(client):
    r = upload_csv(client, "", "empty.csv")
    assert r.status_code == 400
    assert "empty" in r.json()["error"]["message"].lower()


def test_upload_wrong_extension_rejected(client):
    r = client.post(
        "/api/datasets/upload",
        files={"file": ("data.xlsx", b"some bytes", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "csv" in r.json()["error"]["message"].lower()


def test_upload_binary_garbage_rejected(client):
    r = client.post(
        "/api/datasets/upload",
        files={"file": ("garbage.csv", b"\x00\x01\x02\xff\xfe binary", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_upload_header_only_rejected(client):
    r = upload_csv(client, "a,b,c\n", "headeronly.csv")
    assert r.status_code == 400
    assert "no data rows" in r.json()["error"]["message"].lower()


def test_upload_all_empty_columns_rejected(client):
    r = upload_csv(client, "a,b\n,,\n,,\n", "allnan.csv")
    assert r.status_code == 400


def test_get_unknown_dataset_404(client):
    r = client.get("/api/datasets/does-not-exist")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "dataset_not_found"


def test_upload_semicolon_separated_csv(client):
    """Excel-style (UZ/RU/DE locales) semicolon CSVs must be auto-detected."""
    content = "a;b;c\n1;2;3\n4;5;6\n"
    r = upload_csv(client, content, "semicolon.csv")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["columns"] == 3
    assert body["column_names"] == ["a", "b", "c"]


def test_upload_tab_separated_csv(client):
    content = "a\tb\tc\n1\t2\t3\n4\t5\t6\n"
    r = upload_csv(client, content, "tab.csv")
    assert r.status_code == 201, r.text
    assert r.json()["columns"] == 3
