from setuptools import find_packages, setup
from typing import List
HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)-> List[str]:
    '''
    This function willo return the list of requirments
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        [req.replace("\n", "") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements
setup(
    name="MlpROJECT1",
    version="0.0.1",
    author="Abhinav",
    author_email="ankit.abhinav10206@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)