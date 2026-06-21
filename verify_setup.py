#!/usr/bin/env python3
"""Validate Google Cloud geospatial ML environment setup."""

from importlib.metadata import PackageNotFoundError, version

from src.config import load_config

PACKAGE_NAMES = [
    "earthengine-api",
    "google-cloud-aiplatform",
    "google-cloud-storage",
    "geemap",
    "pandas",
    "numpy",
    "pyyaml",
]


def verify_earth_engine(project_id: str) -> bool:
    print("===== Earth Engine =====")
    try:
        import ee

        ee.Initialize(project=project_id)
        image = ee.Image("NASA/NASADEM_HGT/001")
        print(f"SUCCESS: Earth Engine connected. Image ID: {image.id().getInfo()}")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def verify_vertex_ai(project_id: str, location: str) -> bool:
    print("===== Vertex AI =====")
    try:
        import vertexai

        vertexai.init(project=project_id, location=location)
        print(
            f"SUCCESS: Vertex AI initialized "
            f"(project={project_id}, location={location})."
        )
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def verify_cloud_storage(project_id: str) -> bool:
    print("===== Cloud Storage =====")
    try:
        from google.cloud import storage

        client = storage.Client(project=project_id)
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
    config = load_config()
    project_id = config.gcp.project_id
    location = config.gcp.region

    results = {
        "Earth Engine": verify_earth_engine(project_id),
        "Vertex AI": verify_vertex_ai(project_id, location),
        "Cloud Storage": verify_cloud_storage(project_id),
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
