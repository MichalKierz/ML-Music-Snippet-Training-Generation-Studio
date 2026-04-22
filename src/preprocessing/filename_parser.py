from pathlib import Path


def get_filename_stem(file_path: str | Path) -> str:
    return Path(file_path).stem.strip()


def parse_filename_parts(file_path: str | Path, delimiter: str) -> dict:
    stem = get_filename_stem(file_path)

    if delimiter in stem:
        left, right = stem.split(delimiter, 1)
        filename_artist = left.strip()
        filename_title = right.strip()
        split_found = bool(filename_artist or filename_title)
    else:
        filename_artist = ""
        filename_title = ""
        split_found = False

    return {
        "filename_stem": stem,
        "filename_artist": filename_artist,
        "filename_title": filename_title,
        "split_found": split_found,
    }


def resolve_artist_title(file_path: str | Path, delimiter: str, tag_artist: str, tag_title: str) -> dict:
    parsed = parse_filename_parts(file_path, delimiter)

    tag_artist_value = (tag_artist or "").strip()
    tag_title_value = (tag_title or "").strip()
    filename_artist = parsed["filename_artist"]
    filename_title = parsed["filename_title"]
    filename_stem = parsed["filename_stem"]
    split_found = parsed["split_found"]

    if tag_artist_value:
        artist = tag_artist_value
        artist_source = "tag_artist"
    elif filename_artist:
        artist = filename_artist
        artist_source = "filename_before_delimiter"
    else:
        artist = "unknown"
        artist_source = "unknown"

    if tag_title_value:
        title = tag_title_value
        title_source = "tag_title"
    elif tag_artist_value:
        title = filename_stem
        title_source = "filename_full_stem_due_to_tag_artist"
    elif split_found and filename_title:
        title = filename_title
        title_source = "filename_after_delimiter"
    else:
        title = filename_stem
        title_source = "filename_full_stem"

    return {
        "artist": artist,
        "title": title,
        "artist_source": artist_source,
        "title_source": title_source,
        "filename_stem": filename_stem,
        "filename_artist": filename_artist,
        "filename_title": filename_title,
        "split_found": split_found,
    }