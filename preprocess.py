import numpy as np
import pandas as pd
from pathlib import Path
import xlrd
import datetime
import pickle as pkl


root_transcriptome = "platelets.txt"
root_clinical = "class-label.xlsx"


transcriptome_data = pd.read_table(root_transcriptome,
                                   encoding="utf-16",
                                   index_col=0).T

workbook = xlrd.open_workbook(root_clinical)
workbook.sheet_names()
worksheet_ITP = workbook.sheet_by_name('ITP')
worksheet_normal = workbook.sheet_by_name('Control')


def transform_func(item):
    return item.value


def excel_date(item):
    return datetime.datetime(
        *xlrd.xldate_as_tuple(item.value, workbook.datemode))

def age_func(row):
    return float(row["age"])


rows = []
for i in range(worksheet_ITP.nrows):
    row = worksheet_ITP.row(i)
    if i == 0:
        row = list(map(transform_func, row))
    else:
        row = [
            transform_func(row[0]),
            transform_func(row[1]),
            transform_func(row[2])
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
clinical_normal.loc[:, "age"] = clinical_normal.apply(age_func, axis=1)
clinical_data = clinical_ITP.append(clinical_normal).set_index("ID")
clinical_data.loc[:, "gender"] = clinical_data["gender"].astype('category')


transcriptome_data = transcriptome_data.loc[clinical_data.index, :]


with open("data.pkl", "wb") as f:
    pkl.dump(
        {
            "clinical_data": clinical_data,
            "transcriptome_data": transcriptome_data
        }, f)

print("clinical_data: ")
print(clinical_data.shape)


print("transcriptome_data: ")
print(transcriptome_data.shape)
