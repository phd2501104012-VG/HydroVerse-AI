from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="hydroverse-ai",
    version="2.0.0",
    description="Next-Gen Climate Hazard Intelligence Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="HydroVerse AI Team",
    url="https://github.com/hydroverse-ai",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={
        "api": ["fastapi>=0.104.0", "uvicorn>=0.24.0", "websockets>=12.0", "pydantic>=2.0.0"],
        "dl": ["tensorflow>=2.13.0", "torch>=2.0.0"],
        "export": ["openpyxl>=3.1.0", "kaleido>=0.2.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0", "black>=23.0.0", "ruff>=0.1.0"],
    },
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "hydroverse=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: GIS",
    ],
)
