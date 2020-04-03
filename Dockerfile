FROM python:3.6.5-jessie

COPY requirements.txt /
RUN pip install -r /requirements.txt
CMD ["jupyter", "notebook", "--NotebookApp.token='password'", "--ip=0.0.0.0", "--allow-root"]