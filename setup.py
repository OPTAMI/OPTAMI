from setuptools import setup, find_packages

setup(
        name="OPTAMI",
        version="0.0.2",
        author="Dmitry Kamzolov",
        author_email="kamzolov.opt@gmail.com",
        description="Second-Order PyTorch Optimizers",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/OPTAMI/OPTAMI",
        packages=find_packages(),
        python_requires= '>=3.6',
        install_requires=[],
        keywords=['python', 'pytorch', 'optimization', 'optimizer'],
        classifiers= [
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ]
)