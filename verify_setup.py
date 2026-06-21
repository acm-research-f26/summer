#!/usr/bin/env python3
"""Validate Google Cloud geospatial ML environment setup."""

from importlib.metadata import PackageNotFoundError, version

PROJECT_ID = "datacenter-summer-poc"
LOCATION = "us-central1"

PACKAGE_NAMES = [
    "earthengine-api",
    "google-cloud-aiplatform",
    "google-cloud-storage",
    "geemap",
    "pandas",
    "numpy",
]


def verify_earth_engine() -> bool:
    print("===== Earth Engine =====")
    try:
        import ee

        ee.Initialize(project=PROJECT_ID)
        image = ee.Image("NASA/NASADEM_HGT/001")
        print(f"SUCCESS: Earth Engine connected. Image ID: {image.id().getInfo()}")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def verify_vertex_ai() -> bool:
    print("===== Vertex AI =====")
    try:
        import vertexai

        vertexai.init(project=PROJECT_ID, location=LOCATION)
        print(
            f"SUCCESS: Vertex AI initialized "
            f"(project={PROJECT_ID}, location={LOCATION})."
        )
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def verify_cloud_storage() -> bool:
    print("===== Cloud Storage =====")
    try:
        from google.cloud import storage

        client = storage.Client(project=PROJECT_ID)
        bucket_names = [bucket.name for bucket in client.list_buckets()]
        if bucket_names:
            print("SUCCESS: Available buckets:")
            for name in bucket_names:
                print(f"  - {name}")
        else:
            print("SUCCESS: Connected, but no buckets found in this project.")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def verify_package_versions() -> bool:
    print("===== Installed Packages =====")
    all_found = True
    for package in PACKAGE_NAMES:
        try:
            pkg_version = version(package)
            print(f"  {package}: {pkg_version}")
        except PackageNotFoundError:
            all_found = False
            print(f"  {package}: NOT INSTALLED")
        except Exception as exc:
            all_found = False
            print(f"  {package}: FAILED ({exc})")
    if all_found:
        print("SUCCESS: All listed packages are installed.")
    else:
        print("FAILED: One or more packages are missing or could not be queried.")
    return all_found


def main() -> None:
    results = {
        "Earth Engine": verify_earth_engine(),
        "Vertex AI": verify_vertex_ai(),
        "Cloud Storage": verify_cloud_storage(),
        "Installed Packages": verify_package_versions(),
    }

    print("\n===== Summary =====")
    for name, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {name}: {status}")

    if all(results.values()):
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed. Review the output above.")


if __name__ == "__main__":
    main()
