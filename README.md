## how to run 

1. Install the required packages with the provided requirements.txt

   * If you can't install gym==2.21.0,which is necessary for this project,try the following two installation,then the gym will be installed successfully!

     ```python
     pip install setuptools==65.5.0
     pip install --user wheel==0.38.0
     ```

2. Install the VIMABench with [VIMABench](https://github.com/vimalabs/VimaBench).

3. Change the OpenAI API-key in *data_process/gptutils.py*

4. Put the path of SentenseBert in *retrieval/similarity_retirval.py*

5. Put the path of MMLM in *model/custom_model.py*

6. run the *eval.py*.

