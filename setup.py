from setuptools import find_packages, setup

_subpackages = find_packages("src")

setup(
    name="empower",
    version="0.0.1",
    license="MIT",
    package_dir={"empower": "src"},
    packages=["empower"] + [f"empower.{pkg}" for pkg in _subpackages],
    include_package_data=True,
    package_data={
        "empower.yolo_world": [
            "third_party/**/*.py",
            "third_party/**/*.yaml",
            "third_party/**/*.yml",
            "easydeploy/**/*.cpp",
            "easydeploy/**/*.h",
            "easydeploy/**/*.txt",
            "easydeploy/**/*.md",
        ],
    },
)
