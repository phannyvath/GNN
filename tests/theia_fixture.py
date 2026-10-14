"""Explicitly synthetic CDM18 fixture; never describe this as DARPA evidence."""
import csv
import json
from pathlib import Path
import uuid

PREFIX = 'com.bbn.tc.schema.avro.cdm18.'


def node_id(i): return str(uuid.UUID(int=i)).upper()


def make_fixture(directory, per_partition=24):
    directory = Path(directory); directory.mkdir(parents=True,exist_ok=True)
    raw = directory/'SYNTHETIC_cdm18.jsonl'; labels = directory/'SYNTHETIC_labels.csv'
    def datum(kind,value): return {'datum':{PREFIX+kind:value}}
    definitions = []; events = []; truth = []; serial = 10000000
    for part,base in enumerate((100,1100,2100)):
        for j in range(per_partition):
            subject = node_id(1+part*per_partition*2+j*2); obj = node_id(2+part*per_partition*2+j*2)
            definitions.extend([datum('Subject',{'uuid':subject,'type':'SUBJECT_PROCESS'}),
                                datum('FileObject',{'uuid':obj,'type':'FILE_OBJECT_FILE'})])
            malicious = int(j%4==0)
            for n in (subject,obj): truth.append({'uuid':n,'label':malicious,'evidence':'Synthetic fixture label, NOT real ground truth'})
            for k in range(3 if malicious else 1):
                serial += 1
                events.append(datum('Event',{'uuid':node_id(serial),'subject':{PREFIX+'UUID':subject},
                    'predicateObject':{PREFIX+'UUID':obj},'predicateObject2':None,
                    'timestampNanos':base+(j*800//per_partition)+k,'type':'EVENT_WRITE' if malicious else 'EVENT_READ'}))
    # Out-of-order events and late entity definitions exercise disk ordering and resolution.
    with open(raw,'w') as stream:
        for row in events[::-1]+definitions: stream.write(json.dumps(row)+'\n')
    with open(labels,'w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['uuid','label','evidence']);writer.writeheader();writer.writerows(truth)
    return raw,labels
