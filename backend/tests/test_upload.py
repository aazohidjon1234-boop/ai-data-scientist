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


# ------------------------------------------------- separator consistency
SEMICOLON_CSV = (
    "id;age;height;weight;cardio\n"
    + "\n".join(f"{i};{20 + i % 50};{150 + i % 40};{50 + i % 45};{i % 2}" for i in range(40))
    + "\n"
)


def test_semicolon_file_keeps_its_columns_after_upload(client, tmp_path):
    """Excel writes ';' in most non-US locales.

    Detection used to run only on upload, so the file arrived with the right
    column count and every later read returned a single column.
    """
    from app.utils.dataframe_store import read_csv_path
    from app.services.dataset_service import parse_csv

    on_upload = parse_csv(SEMICOLON_CSV.encode())
    path = tmp_path / "semi.csv"
    path.write_text(SEMICOLON_CSV)
    on_reread = read_csv_path(path)

    assert on_upload.shape[1] == 5
    assert list(on_reread.columns) == list(on_upload.columns)
    assert on_reread.shape == on_upload.shape


def test_tab_separated_file_is_detected(tmp_path):
    from app.utils.dataframe_store import read_csv_path

    path = tmp_path / "t.tsv"
    path.write_text("a\tb\tc\n1\t2\t3\n4\t5\t6\n")
    assert read_csv_path(path).shape[1] == 3


def test_plain_comma_file_is_unaffected(tmp_path):
    from app.utils.dataframe_store import read_csv_path

    path = tmp_path / "c.csv"
    path.write_text("a,b\n1,2\n3,4\n")
    df = read_csv_path(path)
    assert list(df.columns) == ["a", "b"] and df.shape == (2, 2)


def test_single_column_file_stays_single(tmp_path):
    """A genuine one-column file must not be split by a stray separator."""
    from app.utils.dataframe_store import read_csv_path

    path = tmp_path / "one.csv"
    path.write_text("note\nhello world\nsecond line\n")
    assert read_csv_path(path).shape[1] == 1


def test_semicolon_dataset_analyses_end_to_end(client):
    """The crash this came from: 0 numeric columns -> correlation skipped."""
    import io

    r = client.post("/api/datasets/upload",
                    files={"file": ("semi.csv", io.BytesIO(SEMICOLON_CSV.encode()), "text/csv")})
    assert r.status_code == 201, r.text
    ds_id = r.json()["id"]
    assert r.json()["columns"] == 5

    a = client.post(f"/api/datasets/{ds_id}/analyze", json={})
    assert a.status_code == 200, a.text
    assert a.json()["profile"]["columns"] == 5
