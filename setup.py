from setuptools import setup, find_packages

setup(
    name="final_project",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gymnasium",
        "stable-baselines3",
        "numpy",
    ],
    description="Custom environments for OpenAI Gym",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
)
