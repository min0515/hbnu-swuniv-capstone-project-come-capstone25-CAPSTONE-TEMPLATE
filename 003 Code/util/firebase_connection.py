import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials
import datetime
from datetime import datetime as dt
from collections import deque

class FirebaseConnection:
    def __init__(self):
        file_path = '/home/dfx/workspace/park/strawberry_detection_final/util/strawberry-detection.json'
        cred = credentials.Certificate(file_path)
        firebase_admin.initialize_app(cred)

        self.db = firestore.client()
        self.doc_data = self.db.collection('Harvest_Data').document('Data')
        self.doc_log = self.db.collection('Growth_Log')
        self.last_log = self.get_last_log()
        self.count = 0
        self.cumul_count = 0

        # 실시간 현황 화면 초기화
    def init_data(self, n_total, n_mature):
        self.doc_data.update({'n_total': n_total,
                            'n_mature': n_mature,
                            'n_immature': n_total - n_mature,
                            'n_harvest': 0})
        
        # 새로운 로그 초기화
    def init_log(self, n_total, n_mature):
        n_cumul_harvest = self.get_cumul_harvest()
        self.doc_log.add({'datetime': dt.now(tz = datetime.timezone.utc),
                          'n_cumul_total': n_total + n_cumul_harvest,
                          'n_current_total': n_total,
                          'n_cumul_mature': n_mature + n_cumul_harvest,
                          'n_current_mature': n_mature,
                          'n_cumul_harvest': n_cumul_harvest,
                          'n_current_harvest': 0})
        
        self.last_log = self.get_last_log()
        self.cumul_count = n_cumul_harvest

        # 만약 수확이 끝나고 다른 구역의 딸기를 추가로 탐지했을 경우
    def update_log(self, n_total, n_mature):
        self.last_log.reference.update({'n_cumul_total': self.last_log.get('n_cumul_total') + n_total,
                                        'n_current_total': self.last_log.get('n_current_total') + n_total,
                                        'n_cumul_mature': self.last_log.get('n_cumul_mature') + n_mature,
                                        'n_current_mature': self.last_log.get('n_current_mature') + n_mature})
        
        # 실시간 현황과 기록중인 로그의 수확량+1
    def increment_harvest_count(self):
        self.count += 1
        self.doc_data.update({'n_harvest': self.count})

        self.cumul_count += 1
        self.last_log.reference.update({'n_cumul_harvest': self.cumul_count,
                                        'n_current_harvest': self.count})
        
        # 실시간 현황 0으로 초기화
    def clear_data(self):
        self.doc_data.update({'n_total': 0,
                            'n_mature': 0,
                            'n_immature': 0,
                            'n_harvest': 0})

    def get_last_log(self):
        docs = self.doc_log.order_by('datetime').stream()
        dq = deque(docs, maxlen = 1)

        if len(dq) == 0:
            return 0
        else:
            return dq[0]
    
    def get_cumul_harvest(self):
        if self.last_log == 0:
            n_cumul_harvest = 0
        else:
            n_cumul_harvest = self.last_log.get('n_cumul_harvest')
            
        return n_cumul_harvest