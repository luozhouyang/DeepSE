import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

requirements = []
with open('requirements.txt', mode='rt', encoding='utf-8') as fin:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        requirements.append(line)

setuptools.setup(
    name="deepse",
    version="0.0.3",
    description="**DeepSE**: **Sentence Embeddings** based on Deep Nerual Networks, designed for **PRODUCTION** enviroment!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/luozhouyang/deepse",
    author="ZhouYang Luo",
    author_email="zhouyang.luo@gmail.com",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=requirements,
    extras_require={

    },
    license="Apache Software License",
    classifiers=(
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    )
)
