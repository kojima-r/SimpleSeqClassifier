# -*- coding: utf-8 -*-
import os
import joblib
import numpy as np
import datetime
import sys
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import csv
import dateutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--resample","-r", action="store_true", help="path")
parser.add_argument("--summary","-s", type=str, default=None, help="postfix")
parser.add_argument("--output_postfix","-o", type=str, default="", help="postfix")
parser.add_argument("--info", type=str, default="dataset_28w/info2.csv", help="postfix")
parser.add_argument("--basepath", type=str, default=None, help="postfix")
parser.add_argument("--output_path", type=str, default=None, help="postfix")
parser.add_argument("--input_postfix", type=str, default=None, help="input postfix")
args = parser.parse_args()


indexes = []
info = []
info_header=[]

#info_path=args.info
info_path="resample_dataset_1hour_28w/info2.csv"

## lstm
base_path="resample_dataset_1hour_28w"
output_base="plot_1h_28w_res/"
prev_before_summary_path="summary.csv"
postfix=""

if args.output_path:
    output_base=args.output_path
if args.basepath:
    base_path=args.basepath
if args.input_postfix:
    postfix=args.input_postfix
    

os.makedirs(output_base,exist_ok=True)
with open(info_path) as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar='"')
    header = next(reader)
    info_header=header
    idx=header.index("患者ID")
    for row in reader:
        indexes.append(row[idx])
        info.append(row)
#
#label,患者ID,出生日,死亡日,転帰,在胎週数(wk),在胎日数,出生体重,M or F,単胎;1、双胎;2、品胎;3,Apgar Score; 1 min,Apgar Score; 5 min,
# NICU退院日,眼科初診日,眼科終了日,眼科終了日と網膜血管の完成時期 (一致;0、不一致;1),眼科終了日と網膜血管の完成時期の一致しない症例,発症 (無;0、有;1、不明;2),発症うたがい日,
# 発症日,治療 (無;0、有;1),初回治療日,追加治療 (無;0、有;1),アバスチン使用 (無;0、有;1),硝子体手術 (無;0、有;1),type1 rop,aprop,非典型的rop,眼科治療に対するコメント　(early;0、apropriate;1、late;2),その他,初回治療内容(PC),初回治療内容(Avastin),初回治療内容(Cryo)
#
gestational_index=info_header.index("在胎日数")
sex_index=info_header.index("M or F")
borth_index=info_header.index("出生日")
dead_index=info_header.index("死亡日")
onset_index=info_header.index("発症日")
cure_index=info_header.index("初回治療日")
prev_before_summary=None
if prev_before_summary_path:
    prev_before_summary={}
    filename=prev_before_summary_path
    with open(filename) as csvfile:
        head=next(csvfile)
        for line in csvfile:
            arr=line.strip().split(",")
            pid = arr[0]
            prev_before = arr[1]
            key=pid+prev_before
            if key not in prev_before_summary:
                prev_before_summary[key]=[]
            prev_before_summary[key].append(arr)

def parse_float(x):
    try:
        if x != "" and x != "nan" and x!="不明":
            return float(x)
        else:
            return np.nan
    except:
        print(x)
        return np.nan


def parse_int(x):
    try:
        if x != "" and x != "nan" and x!="不明":
            return int(float(x))
        else:
            return np.nan
    except:
        print(x)
        return np.nan


def plot_day_line(minx, maxx):
    # miny=np.nanmin(map(np.nanmin,yss))
    # maxy=np.nanmax(map(np.nanmax,yss))
    miny, maxy = plt.ylim()
    dx = datetime.timedelta(days=1)
    day_x = minx
    while day_x < maxx:
        plt.plot([day_x, day_x], [miny, maxy], color=cm.gray(0.5))
        day_x += dx
    return


def plot_day_event_line(day_x, color):
    miny, maxy = plt.ylim()
    plt.plot([day_x, day_x], [miny, maxy], color=color, lw=5)
    return

def process(j):
    # for j in range(3):
    index = indexes[j]
    print(index)
    filename=base_path + "/%010d" % (int(index)) + ".csv"
    if not os.path.exists(filename):
        print("[SKIP not found]",filename)
        return
    st_filename=base_path + "/%010d" % (int(index)) + "_status.csv"
    st_enabled=False
    if os.path.exists(st_filename):
        with open(st_filename) as stfile:
            reader = csv.reader(stfile, delimiter=",", quotechar='"')
            header = next(reader)
            h_idx=header.index("日付")
            zone_idx=header.index("Zone")
            st_idx=header.index("Stage")
            plus_idx=header.index("Plus (無;0、pre-plus;1、plus;2)")
            st_enabled=True
            sxs = []
            sys = [[] for i in range(3)]
            for row in reader:
                t = dateutil.parser.parse(row[h_idx])
                zone = parse_int(row[zone_idx])
                st = parse_int(row[st_idx])
                plus = parse_int(row[plus_idx])
                sxs.append(t)
                sys[0].append(zone)
                sys[1].append(st)
                sys[2].append(plus)
    print(base_path + "/%010d" % (int(index))+postfix + "_data.csv") 
    with open(base_path + "/%010d" % (int(index))+postfix + "_data.csv") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        header = next(reader)
        xs = []
        ys = [[] for i in range(5)]
        #,日付,発症,HR bpm,SpO2 %,RESP /min,修正時体重,修正体重_SD,修正時身長,修正身長_SD,HR bpm.delta,SpO2 %.delta,RESP /min.delta,修正時体重.delta,修正体重_SD.delta,修正時身長.delta,修正身長_SD.delta,在胎日数,M or F,単胎;1、双胎;2、品胎;3,Apgar Score; 1 min,Apgar Score; 5 min
        t_idx=header.index("日付")
        hr_idx=header.index("HR bpm")
        sp_idx=header.index("SpO2 %")
        resp_idx=header.index("RESP /min")
        w_idx=header.index("修正時体重")
        w_sd_idx=header.index("修正体重_SD")
        h_idx=header.index("修正時身長")
        h_sd_idx=header.index("修正身長_SD")
        for row in reader:
            t = dateutil.parser.parse(row[t_idx])
            hr = parse_float(row[hr_idx])
            spo2 = parse_float(row[sp_idx])
            resp = parse_float(row[resp_idx])
            w = parse_float(row[w_idx])
            w_sd = parse_float(row[w_sd_idx])
            #if w_sd > 20 or w_sd < -20:
            #    w_sd = np.nan
            h = parse_float(row[h_idx])
            h_sd = parse_float(row[h_sd_idx])
            #if h_sd > 20 or h_sd < -20:
            #    h_sd = np.nan
            xs.append(t)
            ys[0].append(hr)
            ys[1].append(spo2)
            ys[2].append(resp)
            ys[3].append(w_sd)
            ys[4].append(h_sd)
        # determining ranges of x
        minx = min(xs)
        maxx = max(xs) + datetime.timedelta(days=1)
        minx = minx.replace(hour=0, minute=0, second=0, microsecond=0)
        #borth = info[j][borth_index]
        #if borth != "":
        #    minx = dateutil.parser.parse(borth)
        #dead = info[j][dead_index]
        #if dead != "":
        #    maxx = dateutil.parser.parse(dead)
        ##print("min x: ", minx)
        ##print("max x: ", maxx)
        #
        gestational = info[j][gestational_index]
        sex = info[j][sex_index]
        sex_s= "M" if sex==0 else "F"
        #
        plt.figure(figsize=(48, 32))
        n = 6
        m = 1
        if prev_before_summary is not None:
            n=7
            print("==============================")
            pid="%010d" % (int(index))
            key=pid+"after"
            print(key)
            print(prev_before_summary.keys())
            if key in prev_before_summary:
                print("-------------------")
                vec=prev_before_summary[key]
                plt.subplot(n, 1, m)
                m+=1
                p1=[float(el[4]) for el in vec]
                p2=[float(el[5]) for el in vec]
                p3=[float(el[6]) for el in vec]
                #true_y=int(summary[key][0][2])

                pxs=[minx+datetime.timedelta(days=int(el[3])) for el in vec]
                plt.plot(pxs, p1, label="non ROP", marker="o")
                plt.plot(pxs, p2, label="APROP", marker="o")
                plt.plot(pxs, p3, label="Type1 ROP", marker="o")
                plot_day_line(minx, maxx)
                """
                if true_y==0:
                    l="non ROP patient"
                elif true_y==1:
                    l="APROP patient"
                elif true_y==2:
                    l="Type1 ROP patient"
                """
                plt.title(sex_s+" "+str(gestational)+"day "+l)
                plt.xlim(minx, maxx)
                plt.legend()
            quit()
        print(n,m)
        plt.subplot(n, 1, m)
        m+=1
        if st_enabled:
            plt.plot(sxs, sys[0], label="Zone", marker="o")
            plt.plot(sxs, sys[1], label="Stage", marker="o")
            plt.plot(sxs, sys[2], label="Plus", marker="o")
        plot_day_line(minx, maxx)
        dead = info[j][dead_index]
        if dead != "":
            t = dateutil.parser.parse(dead)
            plot_day_event_line(t, color="red")
        onset = info[j][onset_index]
        if onset != "":
            t = dateutil.parser.parse(onset)
            plot_day_event_line(t, color="blue")
        cure = info[j][cure_index]
        if cure != "":
            t = dateutil.parser.parse(cure)
            plot_day_event_line(t, color="green")
        plt.xlim(minx, maxx)
        if m==1:
            plt.title(sex_s+" "+str(gestational)+"day"+"\n red bar: dead, blue bar: onset, green bar: cure")
        else:
            plt.title("red bar: dead, blue bar: onset, green bar: cure")
        plt.legend()

        def plot_attr(idx, name):
            plt.subplot(n, 1, m + idx)
            plt.plot(xs, ys[idx], label=name)
            plot_day_line(minx, maxx)
            plt.xlim(minx, maxx)
            plt.legend()

        #
        plot_attr(0, "HR")
        plot_attr(1, "SpO2")
        plot_attr(2, "RESP")
        plot_attr(3, "W_sd")
        plot_attr(4, "H_sd")

        #print(output_base+ "%010d" % (int(index)) + ".png")
        plt.savefig(output_base+ "%010d" % (int(index)) + ".png")
        plt.clf()

from multiprocessing import Pool
p = Pool(64)
p.map(process, range(len(indexes)))
p.close()

