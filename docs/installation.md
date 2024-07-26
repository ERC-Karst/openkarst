# Installation

This guide will help you get started with the project by cloning the repository and installing the package locally.  

## Cloning the Repository
The first example requires a properly set up ssh key [(further explanation on GitHub)](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent). Alternatively, one can use HTTPS for cloning the repository if SSH is not an option or if you prefer not to set up an SSH key. Using HTTPS does not require any additional setup beyond having a GitHub account and your credentials.  
Create a new folder for openKARST for example '/openkarst_dev' and navigate to it:

### Cloning with SSH

=== "Unix/macOS"

    ``` bash
    cd openkarst_dev
    git clone git@github.com/ERC-Karst/openkarst.git .
    ```
    
    or when using SSH over the HTTPS port (in case of blocked ports)

    ``` bash
    cd openkarst_dev
    git clone ssh://git@ssh.github.com:443/ERC-Karst/openkarst.git .
    ```    
    

=== "Windows"

    ``` bash
    cd openkarst_dev
    git clone git@github.com:ERC-Karst/openkarst.git .
    ```
    
    or when using SSH over the HTTPS port (in case of blocked ports)

    ``` bash
    cd openkarst_dev
    git clone ssh://git@ssh.github.com:443/ERC-Karst/openkarst.git .
    ```  

### Cloning with HTTPS

=== "Unix/macOS"

    ``` bash
    cd openkarst_dev
    git clone https://github.com/ERC-Karst/openkarst .
    ```

=== "Windows"

    ``` bash
    cd openkarst_dev
    git clone https://github.com/ERC-Karst/openkarst .
    ```
     
## Local Installation

To install the package locally, you have two options: a regular local install and an editable install. If you plan to modify the code, the latter one ("development install") is recommended. A regular install copies the package files to the Python environment's package directory. This method is suitable if you do not intend to modify the code.  
An editable install, also known as a "development install," adds the package directory to Python’s import path without copying files. This allows for immediate testing of any changes made to the code. This method is ideal for development and testing purposes.
### Regular local install

Navigate to the cloned directory, i.e., where the pyproject.toml file is located and run the following command:

=== "Unix/macOS"

    ```bash
    python -m pip install .
    ```
    
    or when working in a virtual environment for example
    
    ```bash
    pyenv exec python -m pip install .
    ```

=== "Windows"

    ``` bash
    py -m pip install . 
    ```


### Editable local install

=== "Unix/macOS"

    ```bash
    python -m pip install -e .
    ```
    or when working in a virtual environment for example
	
    ```bash
    pyenv exec python -m pip install -e .
    ```

=== "Windows"

    ``` bash
    py -m pip install -e . 
    ```

