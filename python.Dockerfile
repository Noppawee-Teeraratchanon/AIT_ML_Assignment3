FROM akraradets/ait-ml-base:2023

RUN pip3 install --upgrade pip
RUN pip3 install ipykernel
RUN pip3 install scikit-learn
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install mlflow
RUN pip3 install seaborn
RUN pip3 install ppscore
RUN pip3 install dash

RUN pip3 install dash[testing]
RUN pip3 install pytest
RUN pip3 install pytest-depends

COPY ./code /root/code

CMD tail -f /dev/null