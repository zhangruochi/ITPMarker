import numpy as np
import pandas as pd
from pathlib import Path
import xlrd
import datetime
import pickle as pkl


root_transcriptome = "platelets.txt"
root_clinical = "20200518-anonymous.xlsx"


transcriptome_data = pd.read_table(root_transcriptome,
                                   encoding="utf-16",
                                   index_col=0).T

workbook = xlrd.open_workbook(root_clinical)
workbook.sheet_names()
worksheet_ITP = workbook.sheet_by_name('ITP')
worksheet_normal = workbook.sheet_by_name('正常人')


def transform_func(item):
    return item.value


def excel_date(item):
    return datetime.datetime(
        *xlrd.xldate_as_tuple(item.value, workbook.datemode))

def age_func(row):
    return float(row["年龄"].strip("岁"))


rows = []
for i in range(worksheet_ITP.nrows):
    row = worksheet_ITP.row(i)
    if i == 0:
        row = list(map(transform_func, row))
    else:
        row = [
            transform_func(row[0]),
            transform_func(row[1]),
            transform_func(row[2]),
            excel_date(row[3]),
            transform_func(row[4])
        ]
    # break
    rows.append(row)

clinical_ITP = pd.DataFrame(rows[1:], columns=rows[0])
rows = []
for i in range(worksheet_normal.nrows):
    row = worksheet_normal.row(i)
    row = list(map(transform_func, row))
    # break
    rows.append(row)

clinical_normal = pd.DataFrame(rows[1:], columns=rows[0])


clinical_normal.loc[:, "年龄"] = clinical_normal.apply(age_func, axis=1)
clinical_data = clinical_ITP.append(clinical_normal).set_index("样本编号")
clinical_data.loc[:, "性别"] = clinical_data["性别"].astype('category')


with open("data.pkl", "wb") as f:
    pkl.dump(
        {
            "clinical_data": clinical_data,
            "transcriptome_data": transcriptome_data
        }, f)
