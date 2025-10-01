from models import VitSam, YoloWorld
from utils.config import *
import spacy
import gensim.downloader as api


class Loader:
    _instance = None

    def __new__(cls, *args):
        if not cls._instance:
            cls._instance = super(Loader, cls).__new__(cls)
            cls._instance.initialize(*args)
        return cls._instance

    def initialize(self, *args):
        self._yolow_model = YoloWorld()
        self._vit_sam_model = VitSam(ENCODER_VITSAM_PATH, DECODER_VITSAM_PATH)
        #self._nlp = spacy.load("en_core_web_sm") 
        #self._wv = api.load('word2vec-google-news-300') 
        

    @property
    def nlp(self):
        return self._nlp
    
    @nlp.setter
    def nlp(self, value):
        self._nlp = value

    @property
    def wv(self):
        return self._wv
    
    @wv.setter
    def wv(self, value):
        self._wv = value

    @property
    def yolow_model(self):
        return self._yolow_model
    
    @yolow_model.setter
    def yolow_model(self, value):
        self._yolow_model = value

    @property
    def vit_sam_model(self):
        return self._vit_sam_model
    
    @vit_sam_model.setter
    def vit_sam_model(self, value):
        self._vit_sam_model = value
        