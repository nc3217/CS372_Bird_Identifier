# Setup Instructions
## 1. Clone the Repository
- Open your terminal or command prompt
- Navigate to the desired directory where you would like the project files to be located
- Clone the repo by running the following line:
```bash
git clone https://github.com/nc3217/CS372_Bird_Identifier.git
cd CS372_Bird_Identifier
```

## 2. Create Python Environment
- For example, you can create a venv environment by running the following lines:
  ```bash
  python3 -m venv .venv
  ```
  For macOS/Linux
```bash
source .venv/bin/activate
```

  For Windows
  ```bash
  .venv\Scripts\Activate.ps1
  ```

  Run this as well:
  ```bash
  pip install --upgrade pip
  ```

# 3. Install Python Dependencies
Run this line: 
```bash
pip install -r requirements.txt
```

# 4. Run Sample_Classifier.py
Now that the requirements have been installed, you can run the bird identifier by doing the following steps:
1. Open the sample_classifier.py file in the src folder
2. Replace the file path in the following line 'test_img = "/content/sample_bird.jpg"' with the file path to your bird image
3. Run the file sample_classifier.py
4. This will output the inputted image, the top 5 results for the predicted bird species, and the predicted probability for each species. 

  

