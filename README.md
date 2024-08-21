1 :Overview
This project develops an AI pipeline for image segmentation and object analysis. The pipeline processes input images to segment, identify, and analyze objects, then outputs a summary table with mapped data for each object.

2 : Structure

project_root/
│
├── data/
│   ├── input_images/               # Directory for input images
│   ├── segmented_objects/          # Directory to save segmented object images
│   └── output/                     # Directory for output images and tables
│
├── models/
│   ├── segmentation_model.py       # Script for segmentation model
│   ├── identification_model.py     # Script for object identification model
│   ├── text_extraction_model.py    # Script for text/data extraction model
│   └── summarization_model.py      # Script for summarization model
│
├── utils/
│   ├── preprocessing.py            # Script for preprocessing functions
│   ├── postprocessing.py           # Script for postprocessing functions
│   ├── data_mapping.py             # Script for data mapping functions
│   └── visualization.py            # Script for visualization functions
│
├── streamlit_app/
│   ├── app.py                      # Main Streamlit application script
│   └── components/                 # Directory for Streamlit components
│
├── tests/
│   ├── test_segmentation.py        # Tests for segmentation
│   ├── test_identification.py      # Tests for identification
│   ├── test_text_extraction.py     # Tests for text extraction
│   └── test_summarization.py       # Tests for summarization
│
├── README.md                       # Project overview and setup instructions
├── requirements.txt                # Required Python packages
└── presentation.pptx               # Presentation slides summarizing the project

3 : Installation 

a) Clone the Repository

git clone <repository_url>
cd project_root


b) Create an virtual Environment

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

c) Install Rrequired Packages

pip install -r requirements.txt

4 : Usage 

a) Prepare input Images
Place your input images in the data/input_images/ directory.

b)Run the Pipeline
python main_script.py

c) Run Streamlit app for interactive use
Streamlit run app.py

5: Output
The processed data will be saved in the data/output/ directory, including:

Annotated Images: Images with segmentation overlays.
Summary Table: CSV file containing summarized data for each object.

6: Presentation

A presentation summarizing the approach, implementation, and results is available as presentation.pptx
