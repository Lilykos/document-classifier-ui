from io import StringIO
import streamlit as st
import nltk

from utils import preprocess, predict

# for NLP requirement
try:
    nltk.download('punkt')
    nltk.download('stopwords')
except:
    pass


def click_callback():
    """
    Callback function for the button. It differentiates between the
    2 options of text and file, and does some sanity checks.
    """
    if txt_input:
        result = predict(
            preprocess(txt_input, do_stop, do_stem, do_punct, do_single)
        )
    if uploaded_file:
        if uploaded_file.name.endswith('.txt'):
            string_io = StringIO(uploaded_file.getvalue().decode("utf-8"))
            txt = string_io.read()
            result = predict(
                preprocess(txt, do_stop, do_stem, do_punct, do_single)
            )
        else:
            result = "Please provide a text file (.txt)."
    if txt_input and uploaded_file:
        result = "Please select one of the two data input options. " \
                 "Maybe there is leftover text in the `Text Input` section? We are working on it!"
    if not (txt_input or uploaded_file):
        result = "Please provide input through one of the available options."

    col2.empty()
    col2.write(result)


# MAIN PAGE ATTRS
# generic stuff: title, page, icons, etc
st.set_page_config(page_title="Document Classifier", page_icon="📚")
st.title('Document Classifier')

# COLUMNS FOR DATA INPUT
# we will be using text/file input so we need to differentiate
# and we will be using tabs for that
tab1, tab2 = st.tabs(["Text Input", "File Input"])

txt_input = tab1.text_area('txt', placeholder='Type text:', label_visibility="collapsed")
uploaded_file = tab2.file_uploader('txt', label_visibility='collapsed')


# OPTIONS AND RESULTS
# 2 columns, one for preprocessing options and run,
# the other will be updated with the results
col1, col2 = st.columns(2)

with col1:
    do_stop = st.checkbox('Remove stopwords')
    do_stem = st.checkbox('Stem words')
    do_punct = st.checkbox('Remove punctuation')
    do_single = st.checkbox('Singularize words')

    st.button(
        label='Predict Class', key='run-classification-btn',
        on_click=click_callback, type="primary"
    )

col2.write("### Prediction Results")
