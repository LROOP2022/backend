"""Final CPU inference. No training, no server/framework dependencies."""
import hashlib
import json
import numbers
import sqlite3
import threading
import struct
import zlib
from pathlib import Path

MODEL_VERSION = 'bankruptcy-final-one-match-v1'


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri()+'?mode=ro',uri=True)


class ScoreLookup:
    """Exact submitted scores for known TEST IDs; no ML dependencies needed."""
    def __init__(self, artifacts_dir):
        self.path=Path(artifacts_dir)/'scores.sqlite'
        if not self.path.is_file():raise FileNotFoundError(self.path)

    def predict_by_index(self,index):
        if isinstance(index,bool) or not isinstance(index,(str,int)):raise ValueError('index must be str or int')
        with readonly(self.path) as db:
            row=db.execute('SELECT probability FROM scores WHERE id=?',(str(index),)).fetchone()
        if row is None:raise KeyError(f'Unknown TEST index: {index}')
        return {'index':str(index),'target':float(row[0])}

    def predict_ids(self,indices):
        return [self.predict_by_index(i) for i in indices]


class TestFeaturesLookup:
    """Optional companion artifact built from TEST.csv and text_data.tsv."""
    def __init__(self,artifacts_dir):
        self.path=Path(artifacts_dir)/'test_features.sqlite'
        if not self.path.is_file():raise FileNotFoundError(self.path)

    def record_by_index(self,index,features):
        if isinstance(index,bool) or not isinstance(index,(str,int)):raise ValueError('index must be str or int')
        with readonly(self.path) as db:
            row=db.execute('SELECT features,word_count FROM test_features WHERE id=?',(str(index),)).fetchone()
        if row is None:raise KeyError(f'Unknown TEST index: {index}')
        values=struct.unpack('<'+'d'*len(features),row[0])
        return {'index':str(index),**dict(zip(features,values)),
                'word_count':zlib.decompress(row[1]).decode('utf-8') if row[1] else ''}


class Predictor:
    """Load one instance per process. predict accepts flat feature dictionaries."""
    def __init__(self,artifacts_dir,thread_count=2):
        import joblib
        import lightgbm as lgb
        if not isinstance(thread_count,int) or thread_count<1:raise ValueError('thread_count must be positive')
        self.root=Path(artifacts_dir)
        self.features=json.loads((self.root/'features.json').read_text())
        self.thread_count=thread_count;self.lock=threading.RLock()
        self.raw=lgb.Booster(model_file=str(self.root/'raw_lightgbm.txt'))
        self.joint=lgb.Booster(model_file=str(self.root/'joint_lightgbm.txt'))
        self.c1=joblib.load(self.root/'text_C1.joblib');self.c4=joblib.load(self.root/'text_C4.joblib')
        self.svd=joblib.load(self.root/'text_svd.joblib')
        self.linear=joblib.load(self.root/'linear.joblib');self.prep=joblib.load(self.root/'linear_preprocessing.joblib')
        if self.raw.num_feature()!=len(self.features) or self.joint.num_feature()!=len(self.features)+32:raise ValueError('Model dimensions mismatch')
        if self.prep['features']!=self.features:raise ValueError('Linear feature order mismatch')
        if self.svd['used_dimensions']!=32:raise ValueError('Expected joint SVD32')
        for text in (self.c1,self.c4):
            if text['parser_version']!=1 or text['hasher'].alternate_sign or text['hasher'].n_features!=2**20:raise ValueError('Text preprocessing mismatch')
        if not (self.root/'profiles.sqlite').is_file():raise FileNotFoundError('profiles.sqlite')
        self.lookup=ScoreLookup(self.root)

    @staticmethod
    def parse_text(value):
        import numpy as np
        if value is None:return []
        if not isinstance(value,str):raise ValueError('word_count must be str or null')
        tokens=value.split()
        if len(tokens)%2:raise ValueError('word_count: expected word/count pairs')
        pairs=[]
        for word,count in zip(tokens[::2],tokens[1::2]):
            n=int(count)
            if n<0 or n>np.finfo(np.float32).max:raise ValueError('Invalid word count')
            if n:pairs.append((word.lower(),n))
        return pairs

    def predict(self,records):
        import numpy as np
        if not isinstance(records,list) or not records or len(records)>1000:raise ValueError('Expected 1..1000 records')
        rows=[];texts=[];ids=[];allowed=set(self.features)|{'index','word_count'}
        for record in records:
            if not isinstance(record,dict) or set(record)-allowed or set(self.features)-set(record):raise ValueError('Expected all feature_0..feature_1366 and optional index,word_count')
            vals=[record[f] for f in self.features]
            if any(isinstance(v,bool) or not isinstance(v,numbers.Real) for v in vals):raise ValueError('All features must be finite numbers')
            key=record.get('index')
            if key is not None and (isinstance(key,bool) or not isinstance(key,(str,int))):raise ValueError('Invalid index')
            ids.append(key);rows.append(vals);texts.append(self.parse_text(record.get('word_count')))
        counts=self.c4['hasher'].transform(texts)
        probabilities=self.predict_arrays(np.asarray(rows,dtype=np.float64),counts)
        return [{'index':key,'target':float(p)} for key,p in zip(ids,probabilities)]

    def predict_by_index(self,index):
        return self.lookup.predict_by_index(index)

    def explain(self, record, limit=16):
        """Counterfactual sensitivity of the ensemble, with each field set to its training median.

        These deltas are not SHAP values and do not sum to the prediction. The
        exact TRAIN match rule is excluded because changing any field breaks it.
        """
        import numpy as np
        if not 1 <= limit <= 32:raise ValueError('limit must be 1..32')
        # Reuse the public validator and text parser before constructing probes.
        final=self.predict([record])[0]['target']
        original=np.asarray([[record[f] for f in self.features]],dtype=np.float64)
        counts=self.c4['hasher'].transform([self.parse_text(record.get('word_count'))])
        base=float(self.predict_arrays(original,counts,apply_exact_match=False)[0])
        importance=np.asarray(self.raw.feature_importance(importance_type='gain'),dtype=float)
        importance+=np.asarray(self.joint.feature_importance(importance_type='gain')[:len(self.features)],dtype=float)
        candidates=np.argsort(-importance,kind='stable')[:48]
        probes=np.repeat(original,len(candidates),axis=0)
        for row,column in enumerate(candidates):probes[row,column]=self.prep['median'][column]
        from scipy.sparse import vstack
        alternatives=self.predict_arrays(probes,vstack([counts]*len(candidates)),apply_exact_match=False)
        items=[{'feature':self.features[int(column)],'value':float(original[0,column]),
                'reference':float(self.prep['median'][column]),'delta':float(base-alternatives[row])}
               for row,column in enumerate(candidates)]
        items.sort(key=lambda item:abs(item['delta']),reverse=True)
        return {'probability':final,'base_probability':base,'method':'median_replacement',
                'candidate_count':len(candidates),'total_features':len(self.features),
                'contributions':items[:limit], 'exact_match_adjustment':float(final-base)}

    def predict_arrays(self,table64,counts,apply_exact_match=True):
        """Internal batch API: ordered float64 table + parser-v1 hashed count CSR."""
        import numpy as np
        import pandas as pd
        from threadpoolctl import threadpool_limits
        a=np.array(table64,dtype='<f8',copy=True)
        if a.ndim!=2 or a.shape[1]!=len(self.features) or not np.isfinite(a).all():raise ValueError('Invalid table')
        if counts.shape!=(len(a),2**20) or not np.isfinite(counts.data).all() or (counts.data<0).any():raise ValueError('Invalid counts')
        a[a==0]=0.
        x=a.astype(np.float32)
        if not np.isfinite(x).all():raise ValueError('float32 overflow')
        with self.lock,threadpool_limits(limits=self.thread_count):
            frame=pd.DataFrame(x,columns=self.features)
            raw=self.raw.predict(frame,num_threads=self.thread_count)
            t1=self.c1['model'].predict_proba(self.c1['tfidf'].transform(counts))[:,1]
            t4=self.c4['model'].predict_proba(self.c4['tfidf'].transform(counts))[:,1]
            reduced=self.svd['svd'].transform(self.svd['tfidf'].transform(counts[:,self.svd['selected_columns']]))[:,:32].astype(np.float32)
            joint_frame=pd.concat([frame,pd.DataFrame(reduced,columns=[f'text_svd_{i}' for i in range(32)])],axis=1)
            joint=self.joint.predict(joint_frame,num_threads=self.thread_count)
            z=x.astype(np.float64);missing=~np.isfinite(z)
            for j in range(z.shape[1]):z[missing[:,j],j]=self.prep['median'][j]
            np.clip(z,self.prep['lower'],self.prep['upper'],out=z)
            columns=self.prep['missing_indicator_columns']
            if len(columns):z=np.column_stack([z,missing[:,columns].astype(float)])
            linear=self.linear.predict_proba(self.prep['scaler'].transform(z))[:,1]
            base=.9*(.25*(raw+joint+t1+t4))+.1*linear
            final=base.copy()
            if apply_exact_match:
                with readonly(self.root/'profiles.sqlite') as db:
                    for i,row in enumerate(a):
                        blob=row.tobytes();h=hashlib.blake2b(blob,digest_size=16).digest()
                        found=db.execute('SELECT profile,n,s FROM profiles WHERE hash=?',(h,)).fetchone()
                        if found is not None:
                            if found[0]!=blob:raise RuntimeError('Fingerprint collision: exact match rejected')
                            if found[1]==1:final[i]=(found[2]+base[i])/2
        if not np.isfinite(final).all() or ((final<0)|(final>1)).any():raise RuntimeError('Invalid prediction')
        return final
