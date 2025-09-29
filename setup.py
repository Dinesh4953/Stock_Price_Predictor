from setuptools import find_packages, setup
from typing import List

def get_requirements() -> List[str]:
    """
    this function will return list of requirements

    """
    requirement_lst : List[str] = []
    try:
        with open("requirements.txt", 'r') as file:
            lines = file.readlines()
            ## Procees each line
            for line in lines:
                requirement = line.strip()
                ## Ignore empty lines and -e.
                if requirement and requirement != '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print('requirements.txt file is not found')
        
    return requirement_lst

print(get_requirements())

setup(
    name= "Stock Price",
    version= '0.0.1',
    author='Dinesh',
    author_email='dinu08642@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements(),
    
)

