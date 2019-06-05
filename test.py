from stanfordcorenlp import StanfordCoreNLP
import logging
import json
import pickle
#%%
class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000, lang='zh'):
        self.nlp = StanfordCoreNLP(host, port=port,timeout=30000, lang=lang)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
                      'pipelineLanguage': 'zh','outputFormat': 'json'}

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    @staticmethod
    def tokens_to_dict(_tokens):
        tokens = defaultdict(dict)

        for token in _tokens:
            tokens[int(token['index'])] = {
                'word': token['word'],
                'lemma': token['lemma'],
                'pos': token['pos'],
                'ner': token['ner']}
        return tokens

#%%
#中文有部分功能不支援，再此僅測試斷詞和POS
def corenlp(text):
    sNLP = StanfordNLP()
    #print("Annotate:", sNLP.annotate(text))
    print("POS:", sNLP.pos(text))
    print("Tokens:", sNLP.word_tokenize(text))
    #print("NER:", sNLP.ner(text))
    #print("Parse:", sNLP.parse(text))
    #print("Dep Parse:", sNLP.dependency_parse(text))
#%%
text = ['銘記六四,在香港維園的燭光晚會上，與往年一樣，同樣有獻花、默哀、致悼詞、誦讀大會宣言、全場演唱民主歌曲、播放「天安門母親」成員錄像講話等環節。現場不乏中國大陸的學生和遊客，然而BBC中文記者要求採訪時，他們大多要求化名、不能露臉，反映他們擔心中國當局打壓的疑慮。就讀香港大學的大陸學生黃同學對BBC中文表示，她對香港持續悼念「六四」感到敬佩，如果不是來香港讀書，也許她對這段歷史也會不聞不問，雖然她認為中國爭取民主十分困難，「但至少有一群人支持當年的學生，我覺得這已很好，令我很感動。」來自成都的曾先生今年44歲，30年前他曾為成都的示威學生在街上送水支持。他帶著自己的太太和11歲的女兒特意來到香港，希望自己的女兒可以認識到真正的歷史。他的女兒說，「我今天是來學習關於國家的歷史的，現在覺得這個國家不比其他國家好。」曾先生還說，香港原本應該是中國通往民主的跳板，但現在反倒受中國影響嚴重，對未來香港民主狀況表示擔憂。
']
corenlp(text)