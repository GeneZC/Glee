# -*- coding: utf-8 -*-

import os
import csv
import collections


Example = collections.namedtuple(
    "Example", 
    (
        "uid", 
        "text_a", 
        "text_b", 
        "label",
    )
)


class DataReader:
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def get_train_examples(self):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "train.tsv")), 
            "train",
        )

    def get_dev_examples(self):
        return self._create_examples(
        self._read_tsv(os.path.join(self.data_dir, "dev.tsv")), 
            "dev",
        )

    def get_test_examples(self):
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "test.tsv")), 
            "test",
        )

    @staticmethod
    def _read_tsv(input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @staticmethod
    def get_label_map():
        """Gets the label map for this data set."""
        raise NotImplementedError()

    @staticmethod
    def _create_examples(lines, set_type):
        """Creates examples."""
        raise NotImplementedError()


class IflytekReader(DataReader):
    """Reader for the Iflytek data set."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {
            "薅羊毛": 11,
            "借贷": 95,
            "违章": 74,
            "工具": 70,
            "高等教育": 58,
            "约会社交": 25,
            "职考": 54,
            "新闻": 34,
            "亲子儿童": 71,
            "魔幻": 12,
            "直播": 49,
            "辅助工具": 24,
            "体育竞技": 19,
            "动作类": 18,
            "休闲益智": 17,
            "中小学": 53,
            "同城服务": 4,
            "银行": 99,
            "棋牌中心": 20,
            "其他": 118,
            "外卖": 108,
            "办公": 113,
            "股票": 94,
            "论坛圈子": 28,
            "音乐": 48,
            "理财": 96,
            "经营": 116,
            "MOBA": 23,
            "策略": 22,
            "经营养成": 21,
            "摄影修图": 102,
            "仙侠": 13,
            "彩票": 97,
            "英语": 56,
            "地图导航": 1,
            "视频": 46,
            "小说": 36,
            "问诊挂号": 83,
            "购物咨询": 111,
            "情侣社交": 30,
            "电子产品": 82,
            "百科": 42,
            "射击游戏": 16,
            "收款": 117,
            "打车": 0,
            "母婴": 72,
            "体育咨讯": 90,
            "短视频": 47,
            "漫画": 35,
            "记账": 98,
            "装修家居": 81,
            "政务": 9,
            "成人教育": 59,
            "支付": 92,
            "运动健身": 91,
            "美颜": 100,
            "租房": 79,
            "社区服务": 10,
            "婚恋社交": 29,
            "公共交通": 8,
            "社区超市": 110,
            "兼职": 45,
            "相机": 103,
            "快递物流": 5,
            "菜谱": 88,
            "酒店": 66,
            "影像剪辑": 101,
            "租车": 3,
            "影视娱乐": 43,
            "问答交流": 39,
            "艺术": 60,
            "卡牌": 14,
            "旅游资讯": 62,
            "餐饮店": 89,
            "电商": 106,
            "杂志": 41,
            "医疗服务": 85,
            "二手": 105,
            "教辅": 38,
            "社交工具": 31,
            "团购": 107,
            "行车辅助": 78,
            "汽车交易": 76,
            "绘画": 104,
            "即时通讯": 26,
            "驾校": 73,
            "养生保健": 84,
            "电台": 50,
            "求职": 44,
            "铁路": 65,
            "日程管理": 114,
            "搞笑": 40,
            "语言(非英语)": 61,
            "家政": 7,
            "笔记": 112,
            "免费WIFI": 2,
            "女性": 115,
            "汽车咨询": 75,
            "微博博客": 33,
            "技术": 37,
            "成人": 52,
            "保险": 93,
            "买房": 80,
            "美妆美业": 87,
            "K歌": 51,
            "日常养车": 77,
            "工作社交": 27,
            "飞行空战": 15,
            "电影票务": 109,
            "民航": 64,
            "综合预定": 63,
            "公务员": 55,
            "生活社交": 32,
            "减肥瘦身": 86,
            "行程管理": 67,
            "视频教育": 57,
            "婚庆": 6,
            "出国": 69,
            "民宿短租": 68
        }        
        return lambda x: d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                Example(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=None, 
                    label=label
                )
            )
        return examples

class CMIDReader(DataReader):
    """Reader for the CMID data set."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {
            "病症治疗方法": 0,
            "病症定义": 1,
            "病症临床表现(病症表现)": 2,
            "药物适用症": 3,
            "其他无法确定": 4,
            "病症禁忌": 5,
            "病症相关病症": 6,
            "其他对比": 7,
            "药物副作用": 8,
            "药物禁忌": 9,
            "其他多问": 10,
            "病症病因": 11,
            "治疗方案化验/体检方案": 12,
            "治疗方案恢复": 13,
            "病症严重性": 14,
            "病症治愈率": 15,
            "药物用法": 16,
            "药物作用": 17,
            "其他两性": 18,
            "治疗方案正常指标": 19,
            "其他养生": 20,
            "治疗方案方法": 21,
            "病症传染性": 22,
            "药物成分": 23,
            "病症预防": 24,
            "治疗方案恢复时间": 25,
            "病症推荐医院": 26,
            "治疗方案费用": 27,
            "治疗方案临床意义/检查目的": 28,
            "其他设备用法": 29,
            "治疗方案疗效": 30,
            "药物价钱": 31,
            "治疗方案有效时间": 32,
            "其他整容": 33,
            "病症所属科室": 34,
            "治疗方案治疗时间": 35
        }       
        return lambda x: d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                Example(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=None, 
                    label=label
                )
            )
        return examples

class MSRAReader(DataReader):
    """Reader for the MSRA data set."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {
            "INTEGER": 0,
            "ORDINAL": 1,
            "LOCATION": 2,
            "DATE": 3,
            "ORGANIZATION": 4,
            "PERSON": 5,
            "MONEY": 6,
            "DURATION": 7,
            "TIME": 8,
            "LENGTH": 9,
            "AGE": 10,
            "FREQUENCY": 11,
            "ANGLE": 12,
            "PHONE": 13,
            "PERCENT": 14,
            "FRACTION": 15,
            "WEIGHT": 16,
            "AREA": 17,
            "CAPACTITY": 18,
            "DECIMAL": 19,
            "MEASURE": 20,
            "SPEED": 21,
            "TEMPERATURE": 22,
            "POSTALCODE": 23,
            "RATE": 24,
            "WWW": 25
        }       
        return lambda x: d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                Example(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=text_b, 
                    label=label
                )
            )
        return examples

class CTCReader(DataReader):
    """Reader for the CTC data set."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {
            "Therapy or Surgery": 0,
            "Sign": 1,
            "Addictive Behavior": 2,
            "Age": 3,
            "Disease": 4,
            "Multiple": 5,
            "Organ or Tissue Status": 6,
            "Allergy Intolerance": 7,
            "Compliance with Protocol": 8,
            "Risk Assessment": 9,
            "Pregnancy-related Activity": 10,
            "Diagnostic": 11,
            "Laboratory Examinations": 12,
            "Consent": 13,
            "Blood Donation": 14,
            "Enrollment in other studies": 15,
            "Pharmaceutical Substance or Drug": 16,
            "Capacity": 17,
            "Diet": 18,
            "Special Patient Characteristic": 19,
            "Non-Neoplasm Disease Stage": 20,
            "Researcher Decision": 21,
            "Data Accessible": 22,
            "Life Expectancy": 23,
            "Neoplasm Status": 24,
            "Literacy": 25,
            "Encounter": 26,
            "Exercise": 27,
            "Symptom": 28,
            "Receptor Status": 29,
            "Oral related": 30,
            "Ethnicity": 31,
            "Healthy": 32,
            "Disabilities": 33,
            "Device": 34,
            "Gender": 35,
            "Smoking Status": 36,
            "Sexual related": 37,
            "Nursing": 38,
            "Alcohol Consumer": 39,
            "Address": 40,
            "Education": 41,
            "Bedtime": 42,
            "Ethical Audit": 43
        }       
        return lambda x: d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, i)
            text_a = line[0]
            #text_b = line[1]
            label = line[1]
            examples.append(
                Example(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=None, 
                    label=label
                )
            )
        return examples

class EComReader(DataReader):
    """Reader for the ECommerce data set."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {
            "Negative": 0,
            "Positive": 1
        }       
        return lambda x: d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, i)
            text_a = line[0]
            #text_b = line[1]
            label = line[1]
            examples.append(
                Example(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=None, 
                    label=label
                )
            )
        return examples


class RTEReader(DataReader):
    """Reader for the RTE data set."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {
            "not_entailment": 0,
            "entailment": 1
        }
        return lambda x: d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                Example(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=text_b, 
                    label=label
                )
            )
        return examples

class BoolQReader(DataReader):
    """Reader for the BoolQ data set."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {
            "False": 0,
            "True": 1
        }
        return lambda x: d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                Example(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=text_b, 
                    label=label
                )
            )
        return examples

class R52Reader(DataReader):
    """Reader for the R52 data set."""
    def __init__(self, data_dir):
        super().__init__(data_dir)

    @staticmethod
    def get_label_map():
        d = {
            "copper": 0,
            "livestock": 1,
            "gold": 2,
            "money-fx": 3,
            "tea": 4,
            "ipi": 5,
            "trade": 6,
            "cocoa": 7,
            "iron-steel": 8,
            "reserves": 9,
            "zinc": 10,
            "nickel": 11,
            "ship": 12,
            "cotton": 13,
            "platinum": 14,
            "alum": 15,
            "strategic-metal": 16,
            "instal-debt": 17,
            "lead": 18,
            "housing": 19,
            "gnp": 20,
            "sugar": 21,
            "rubber": 22,
            "dlr": 23,
            "tin": 24,
            "interest": 25,
            "income": 26,
            "crude": 27,
            "coffee": 28,
            "jobs": 29,
            "meal-feed": 30,
            "lei": 31,
            "lumber": 32,
            "gas": 33,
            "nat-gas": 34,
            "veg-oil": 35,
            "orange": 36,
            "heat": 37,
            "wpi": 38,
            "cpi": 39,
            "earn": 40,
            "jet": 41,
            "potato": 42,
            "bop": 43,
            "money-supply": 44,
            "carcass": 45,
            "acq": 46,
            "pet-chem": 47,
            "grain": 48,       
            "fuel": 49,
            "retail": 50,
            "cpu": 51
        }        
        return lambda x: d[x], len(d)

    @staticmethod
    def _create_examples(lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            uid = "%s-%s" % (set_type, i)
            text_a = line[0]
            #text_b = line[1]
            label = line[1]
            examples.append(
                Example(
                    uid=uid, 
                    text_a=text_a, 
                    text_b=None,
                    label=label
                )
            )
        return examples


