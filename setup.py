from setuptools import setup

setup(
    name="medicaNLP",
    version="1.0.0",
    description="Python package to explore clinical information extraction in German",
    packages=[
        "medicaNLP",
        "medicaNLP.cardiode",
        "medicaNLP.bronco",
        "medicaNLP.gernermed",
        "medicaNLP.ggponc",
        "medicalNLP.instructionTuning"
    ]
)
