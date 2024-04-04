import torch
from torch import tensor as T
import pickle
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from indexers import DenseFlatIndexer
from encoder import KobertBiEncoder
from dpr_data import KorQuadDataset, KorQuadSampler, korquad_collator
from utils import get_passage_file
from typing import List


class KorDPRRetriever:
    def __init__(self, model, valid_dataset, index, val_batch_size: int = 64, device='cuda:1' if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = valid_dataset.tokenizer
        self.val_batch_size = val_batch_size
        self.valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset.dataset,
            batch_sampler=KorQuadSampler(
                valid_dataset.dataset, batch_size=val_batch_size, drop_last=False
            ),
            collate_fn=lambda x: korquad_collator(
                x, padding_value=valid_dataset.pad_token_id
            ),
            num_workers=4,
        )
        self.index = index

    def val_top_k_acc(self, k:List[int]=[5] + list(range(10,101,10))):
        '''validation set에서 top k 정확도를 계산합니다.'''
        
        self.model.eval()  # 평가 모드
        k_max = max(k)  # 100
        sample_cnt = 0
        retr_cnt = defaultdict(int)
        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc='valid'):
                q, q_mask, p_id, a, a_mask = batch  # (batch_q, batch_q_attn_mask, batch_p_id, batch_p, batch_p_attn_mask) : p, p_attn 지움
                q, q_mask = (
                    q.to(self.device),
                    q_mask.to(self.device),
                )
                q_emb = self.model(q, q_mask, "query")  # bsz x bert_dim
                result = self.index.search_knn(query_vectors=q_emb.cpu().numpy(), top_docs=k_max)  # 100자로 똑같이 잘린 chunk의 index span에서 k개를 가져옴
                
                for ((pred_idx_lst, _), true_idx, _a , _a_mask) in zip(result, p_id, a, a_mask):
                    a_len = _a_mask.sum()
                    _a = _a[:a_len]
                    _a = _a[1:-1]
                    _a_txt = self.tokenizer.decode(_a).strip()
                    docs = [pickle.load(open(get_passage_file([idx]),'rb'))[idx] for idx in pred_idx_lst]  # to_save = {idx + i: ret[i] for i in range(len(ret))}
                    # print(_a_txt)

                    for _k in k:
                        search_title = _a_txt.split("[SEP]")[0].split("[CLS]")[1].strip()  # 앞에 타이틀만 check
                        # print(search_title)
                        # print(docs[:5])
                        if search_title in ' '.join(docs[:_k]): 
                            retr_cnt[_k] += 1  # answer span이 포함된 passage가 있는지 체크합니다 ? 
                    # eval 뽑아보기
                    # print(retr_cnt)

                bsz = q.size(0)
                sample_cnt += bsz
                _retr_acc = {_k:float(v) / float(sample_cnt) for _k,v in retr_cnt.items()}
                print(f"Accuracy by batch: {_retr_acc}")
        
        retr_acc = {_k:float(v) / float(sample_cnt) for _k,v in retr_cnt.items()}
        return retr_acc


    def retrieve(self, query: str, k: int = 100):
        """주어진 쿼리에 대해 가장 유사도가 높은 passage를 반환합니다."""
        self.model.eval()  # 평가 모드
        tok = self.tokenizer.batch_encode_plus([query])
        with torch.no_grad():
            out = self.model(T(tok["input_ids"]).to(self.device), T(tok["attention_mask"]).to(self.device), "query")  # all tensors to be on the same device : .to(self.device)
        result = self.index.search_knn(query_vectors=out.cpu().numpy(), top_docs=k)

        # 원문 가져오기
        passages = []
        for idx, sim in zip(*result[0]):
            path = get_passage_file([idx])
            if not path:
                print(f"No single passage path for {idx}")
                continue
            with open(path, "rb") as f:
                passage_dict = pickle.load(f)
            print(f"passage : {passage_dict[idx]}, sim : {sim}")
            passages.append((passage_dict[idx], sim))
        return passages


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, required=True)
    parser.add_argument("--k", "-k", type=int, required=True)
    args = parser.parse_args()

    model = KobertBiEncoder()
    model.load("model/1000_iter.pt")  # "model/kobert_biencoder.pt" / "model/my_model.pt"
    model.eval()
    valid_dataset = KorQuadDataset("dataset/KorQuAD_v1.0_dev.json")
    index = DenseFlatIndexer()
    index.deserialize(path="index")
    retriever = KorDPRRetriever(model=model, valid_dataset=valid_dataset, index=index)
    # retriever.retrieve(query=args.query, k=args.k)
    """
    chunk 200
    passage : [CLS] 독일 연방의회[SEP]회 의원의 업무는 회기주와 비회기주의 업무로 나뉜다. 독일 연방의회는 격주로 회의를 개최하며, 법정휴일이 있는 경우 이 주기는 약간 조정된다. 회의가 열리는 주의 활동. 연방의회에서 회의가 있는 주의 하원의원들의 활동은 월요일부터 시작된다. 늦은 오후 원내 대표단과 교섭단체내 주요 위원회, 그 주에 있을 상임위원회와 본회의 준비, sim : 39.508270263671875
    passage : [CLS] 독일 연방의회[SEP]를 신임안과 연계시키기도 했다. 국가기관의 탄핵. 연방의회과 연방참의원, 연방대통령이 헌법이나 연방법을 의도적으로 위반한 경우, 헌법재판소에 탄핵 심판을 요구할 수 있다. 양원 중 하나에서 2/3의 찬성을 통해 탄핵을 발의할 수 있다. 연방회의에서 대통령이 새롭게 선출되고, 임무를 더 이상 수행할 수 없게 되면, 탄핵안은 종결된다. 하지만 의회, sim : 39.47883987426758
    passage : [CLS] 기토 히로시[SEP] 않았으며 김일융은 1968년 여름 고시엔 대회 준우승 뒤 다이요 외에도 샌프란시스코 자이언츠 난카이 한큐 도에이 니시테쓰 히로시마 주니치 등에서 입단 교섭이 왔지만 요미우리를 선택했고 오리온스는 오노 쇼이치 미히라 하루키 이후 쓸만한 좌완투수 보강을 위해 영입한 본인(기토)가 4승(3선발), sim : 38.89962387084961
    passage : [CLS] 착생식물[SEP]키에르 체계의 하위분류 중 하나다. 착생식물"epiphytic"이라는 용어는 그리스어의 '위에'를 의미하는 "epi-", '식물'을 의미하는 "phyton"에서 유래하였다. 착생식물은 땅에 뿌리를 뻗지 않기 때문에 때때로 "공기식물air plants"이라, sim : 38.373199462890625
    passage : [CLS] 착생식물[SEP]9%, 24,000여종이 현화식물이다. 두 번째로 큰 분류집단은 박낭성 양치류이며 10%, 2,800여종이 존재한다. 사실 모든 양치식물의 3분의 1이 착생식물이다. 세 번째로 큰 분류집단은 190종이 존재하는이며 각각 수십개의 부처손, 다른 양치식물, 네타속, 소철 등이 이를 뒤따른다. 착, sim : 37.820037841796875
    passage : [CLS] 중국의 고전극[SEP] &lt;살구기([UNK])&gt;, 작자불명의 &lt;유지원([UNK]知[UNK])&gt;([UNK])을 명대 초기의 4대 전기라 하고 중기(中[UNK])에서는 왕구사(王[UNK])의 &lt;도보유춘([UNK])&gt;, 양진어의 &lt;완사기([UNK])&gt;가 저명하나, sim : 36.63426208496094
    passage : [CLS] 열대흙파는쥐[SEP][CLS] 열대흙파는쥐 또는 열대주머니고퍼("Geomys tropicalis")는 흙파는쥐과에 속하는 설치류의 일종이다. 멕시코의 토착종이다. 자연 서식지는 무더운 사막 지역이다. 서식지 감소로 위협을 받고 있다. 특징. 열대흙파는쥐는 등과 머리는 황갈색과 갈색을 띤다., sim : 36.61955261230469
    passage : [CLS] 야세르 아라파트[SEP] 국제적으로 비난을 받았다. 아라파트는 공격들로부터 자신과 팔레스타인 해방 기구를 공개적으로 분리하였다. 그 동안 이스라엘의 골다 메이어 총리는 유럽에서 운영되는 파타의 세포들을 무너뜨리는 데 "바요네트 작전"으로 불린 캠페인을 정식으로 허가하였다. 1973년 ~ 1974년 아라파트는 해외 공격들이 너무 나쁜 홍보를 끌어들였기 때문에, sim : 36.20732498168945
    passage : [CLS] 기토 히로시[SEP]에 그치자 다음 해 다이요로 팔아버렸으며 1980년 미즈타니 노리히로(1973년 시즌 중 주니치에서 이적) (10선발승) 이전까지 한동안 두자릿수 선발승을 거둔 좌완투수가 전무했었다. 선수로서의 특징. 스리쿼터에서 커브, 포크볼, 슈토를 무기로 삼았다.[SEP], sim : 36.12450408935547
    passage : [CLS] 벌거숭이 임금님[SEP]을 만난 자리에서 근사한 옷을 지어주겠다고 하였으나, 이들이 지어준 옷은 "눈에 보이지 않는 옷"이었다. 그렇지만 임금님의 눈에 보이지 않는다는 옷은 실제로는 아무것도 없는 옷이었다. 임금님이 이 옷을 입고 길거리를 행진하자 사람들은 처음에는 임금님을 칭송하면서도 본인들도 덩달아 바보 소리를 듣고 싶지 않아서 일부러 함구하고 있었다. 그런데 한 아이, sim : 36.110233306884766
    passage : [CLS] 착생식물[SEP] 많은 종들에게 중요한 먹이의 원천이다. 착생식물은 물리적인 지지를 위해 다른 식물 위에서 자라며, 필연적으로 숙주에게 부정적인 영향을 끼치지 않기 때문에 기생생물과는 다르다. 식물 외의 착생하는 생물은 때로 착생생물이라고 부른다. 착생식물은 대개 온화한 지역(예를 들어 수많은 이끼, 우산이끼, 지의류,, sim : 36.068756103515625
    passage : [CLS] 표도르 튜틴[SEP]였다. 2006-07 시즌에는 66경기에 출전하여 2골을 넣고 12개의 어시스트를 기록하였다. 2008년 2월 17일에 1137만 5천 미국 달러의 연봉으로 4년 연장 계약을 하였고 2007-08 시즌에 82경기에 출전하여 5골을 넣고 15개의 어시스트를 기록하였다. 2008년 7월 2일 팀 동료인 크리스티안 베크만과 함께 NHL 클럽인 콜럼버스 블루, sim : 36.0078239440918
    passage : [CLS] 압델카림 하산[SEP][CLS] 압델카림 하산(, Abdelkarim Hassan, 1993년 8월 28일 ~ )은 카타르의 축구 선수이다. 포지션은 수비수로 현재 이란의 페르세폴리스 소속이다. 선수 경력. 2011년 카타르의 아스파이어 아카데미를 졸업했다. AFC 챔피언스리그 2011에 17살의 나이로 참가를 했으며, 이 시즌에서 소속, sim : 35.96316146850586
    passage : [CLS] 아시카가 요시미쓰[SEP] 시도했으나, 이것도 천황의 가신과의 교섭은 할 수 없다는 이유와 국서의 수신인을 승상으로 했다는 이유로 입공을 거절당했다. 그래서 요시미쓰는 1394년 12월에 태정대신을 사직하고 출가했다. 이로써 요시미쓰는 천황의 신하가 아닌 자유로운 입장이 되었다. 1401년(오에이 8년), '일본국 준삼후, sim : 35.76514434814453
    passage : [CLS] 부산광역시[SEP]텀시티가 있다. 그 밖에 동래구 온천동에 롯데백화점 동래점, 동구 범일동에 현대백화점 부산점 등이 있다. 언론 기관. 부산과 경상남도 전역을 방송권역으로 하는 민영 방송사인 KNN 부산경남방송을 비롯한, 공영 방송사인 KBS부산방송총국, 대한민국 최초의 민영 방송사인 부산문화방송이 주요 방송사이다. 그 밖에 부산CBS, f, sim : 35.73915481567383
    passage : [CLS] 착생식물[SEP][CLS] 착생식물(Epiphyte)은 식물의 표면에서 자라나며, 필요한 영양분을 공기, 강우, 해양 환경의 경우 물, 주변에 쌓인 잔해물에서 흡수하는 식물이다. 착생식물은 영양순환에 참여하며, 다른 유기체처럼 착생식물이 존재하는 생태계의 다양성과 생물량에 기여한다. 착생식물은, sim : 35.62813186645508
    passage : [CLS] 왜성대[SEP][CLS] 왜성대([UNK])는 조선시대의 지명으로 현재 서울특별시 중구 예장동·회현동1가에 걸쳐 있던 지역이다. 개요. 임진왜란 때 왜군들이 주둔한 데서 마을이름이 유래되었다. 이곳에 조선시대에 군사들이 무예를 연습하던 훈련장인 무예장이 있었으므로 무예장을 줄여 예장 혹은 예장골이라 하였다. 이후 1885년 도, sim : 35.57680892944336
    passage : [CLS] 긴테쓰 특급사[SEP], 1942년 11월부터는 준급의 운행 속도가 낮춰진데 이어, 1943년 2월에는 아예 준급 열차의 운행이 중단되었다. 태평양 전쟁이 끝난 이후, 이 구간에는 우등 열차는 운행하지 않고, 보통 열차만이 다수 운행했다. 그나마도 1964년 10월부터는 미나토마치 역에서 이세시마까지 직통하는 보통 열차마저 사라졌다. 나고야 - 이세 구간., sim : 35.43846893310547
    passage : [CLS] 2017년 동계 아시안 게임 쇼트트랙[SEP]리아와 뉴질랜드는 초청국 자격으로 경기에 참여하기 때문에 입상을 하여도 메달이 주어지지 않는다. 괄호 안은 참가 선수의 숫자이다.[SEP], sim : 35.384422302246094
    passage : [CLS] 홍무제[SEP]았다. '정난의 변'을 일으켜 1402년에 조카 건문제를 폐위 시키고 즉위한 영락제는 환관을 중용했기 때문이다. 영락제에 의해 시작된 환관 정치는 명황조 말기까지 존속되었다. 또한 6대 정통제의 총신인 환관 왕진은 정권을 잡은직후 환관의 정치 간여금지를 천명한 철패를 부셔버렸다., sim : 35.275367736816406
    --------------------------------------------
    chunk 100
    passage : [CLS] 주일본 네덜란드 대사관[SEP][CLS] 주일본 네덜란드 대사관(, )은 도쿄도 미나토구 시바고엔 3-6-3에 위치한 네덜란드 대사관이다. 현직 주일본 네덜란드 대사는 현재 공석 상태이다. 대사. 2023년 8월 10일자로, 시어도러스 페터스가 임시대리대사로 임명되었다. 특이 사항. 주한 네덜란드 대사관이 개설되기 직전부터 정식 주, sim : 41.8963508605957
    passage : [CLS] 피어스 [UNK]거[SEP][CLS] 피어스 [UNK]거 (Piers Wenger, 1972년 6월 29일~)는 BBC 드라마 기획본부장을 지내고 있는 영국의 방송인이다. 경력. 1972년 6월 29일 영국 스태퍼드셔주의 스토크온트렌트에서 피어스 존 [UNK]거 (Piers John Wenger)라는 이름으로 태어났다. 2000년대 초부터 프로듀서로 활동, sim : 41.86698913574219
    passage : [CLS] Emerson, Lake &amp; Palmer (음반)[SEP]티스트는 록팝의 마이크 골드스타인과의 인터뷰에서 이를 부인했다. "위키피디아에 따르면, 그 이미지가 LA 밴드 스피릿과 어떻게든 연관되어 있다는 소문을 잠시나마 불식시키고 싶습니다. 사실은 ELP "새"를 그릴 당시 LA에서 그들에게 보낸 스피릿의 초상화도 그렸습니다. 그 그림의 구석에는 아주 비슷한 새가 등장했습니다., sim : 41.66953659057617
    passage : [CLS] 구드룬 쉬만[SEP]. 남성세를 비롯한 그의 각종 정책들은 많은 논란의 대상이 되었다.[SEP], sim : 41.56045913696289
    passage : [CLS] 페드로 콘트레라스[SEP]되었다. 클럽 경력. 마드리드 출신으로 레알 마드리드 유소년부를 졸업한 콘트레라스는 1998년에 UEFA 챔피언스리그를 우승한 선수단에 포함되었지만, 그 해의 유럽대항전에 1분도 출전하지 못했다. 그는 4년 동안 2군인 레알 마드리드 카스티야에서 보내면서, 4차례 1998-99 시즌 1군 경기에 참가하였고, 소속 구단은 준우승을 거두었다. 1996-97 시즌에는 라, sim : 40.88275909423828
    passage : [CLS] 페드로 콘트레라스[SEP]요 바예카노에 임대되어 활약했는데, 42번의 리그 경기 중 단 1번만 결장했지만, 소속 구단은 라 리가에서 강등당했다. 콘트레라스는 1999-2000 시즌에 말라가로 이적하였다. 그 곳에서, 그는 붙박이 주전으로 활약하며 1999년부터 2003년까지 1부 리그 경기를 6번만 결장했다. 2003년 여름, 콘트레라스는 베티스로 이적, sim : 40.428993225097656
    passage : [CLS] 피어스 [UNK]거[SEP]하였다. 이 시기 영국의 또다른 프로듀서 빅토리아 우드와 주요 작품제작에서 10년 넘게 긴밀히 협력해 왔다. 2006년 ITV에서 방영된 &lt;주부 49세&gt;의 프로듀서를 맡아 영국 아카데미상과 RTA상을 수상하였으며, 2007년 &lt;발레 슈즈&gt;의 프로듀서를 맡았다. 2009년 BBC 웨, sim : 40.36759567260742
    passage : [CLS] 럭키맨[SEP][CLS] 럭키맨()은 다음을 가리킨다.[SEP], sim : 40.149024963378906
    passage : [CLS] 주일본 네덜란드 대사관[SEP]한 네덜란드 대사가 임명되기 전까지 이 대사관에 소속된 대사가 주한 대사까지 겸임한 이력이 과거에 있었다.[SEP], sim : 40.05335235595703
    passage : [CLS] 후티-사우디아라비아 분쟁[SEP][CLS] 후티-사우디아라비아 분쟁은 예멘의 후티 반군이 2015년부터 현재까지 사우디아라비아 남부 국경을 침략하여 발생한 분쟁이다. 역사. 2015년 1월, 후티 반란이 성공해, 친이란계 후티 반군이 친사우디아라비아계 예멘 수도의 대통령궁을 장악했다. 2015년 4월 2일 후티 반군이 사우디아라비아 남부, sim : 39.609657287597656
    passage : [CLS] 간노의 소란[SEP] 있었다. 원래 타카우지의 아버지 사다우지는 호조 씨 소생의 적남이었던 다카요시(高[UNK])에게 가독을 남겨주고 가재인 고노 모로시게(高[UNK]重, )에게 다카요시 보좌를 명했지만 다카요시가 요절하는 바람에 우에스기 씨 소생으로서 다카요시의 이복동생으로 태어난 타카우지가 후, sim : 38.90449523925781
    passage : [CLS] 정종덕[SEP][CLS] 정종덕([UNK], 1943년 ~ 2016년 2월 16일)은 대한민국의 전 축구 감독이며, 감독으로서 카리스마 넘치는 지도력을 바탕으로 발전 가능성이 있는 선수들을 발굴하는 안목과 선수들을 조련해 성장시키는 능력에 능했던 것으로 평가되었다. 또한 건국대학교 감독 재임 시절 황선홍, 유상철, 이영표 등 여러 스타 선수들을 지도했으며, 특히 스트라이커였던 유상철을 수비형 미드필더, sim : 38.512176513671875
    passage : [CLS] 후티-사우디아라비아 분쟁[SEP] 국경선을 쳐들어왔다. 4년 넘게 전쟁중이다. 2016년 8월 16일, 예멘과 인접한 사우디의 나즈란 시에 후티 반군이 포격을 가해 민간인 7명이 사망했다.[SEP], sim : 38.288002014160156
    passage : [CLS] 브라밤 오토모티브[SEP][CLS] 브라밤 오토모티브()는 2018년 5월에 데이비트 브라밤과 오스트레일리아의 투자자 그룹인 퓨전 캐피탈이 출범한 오스트레일리아의 자동차 제조업체로, 본사는 애들레이드에 있다.[SEP], sim : 38.259544372558594
    passage : [CLS] 페드로 콘트레라스[SEP]토니오 카마초 감독에 의해 뒤늦게 차출되었다. 그가 출전한 유일한 국가대표팀 경기는 0-0으로 비긴 파라과이와의 같은 해 10월 16일 경기로, 스페인은 로그로뇨 안방에서 득점 없이 비겼다.[SEP], sim : 38.201133728027344
    passage : [CLS] 대한민국 제16대 국회 후반기 의장단 선거[SEP] 등, 선거는 한나라당 측에 유리하게 진행되었다. 선거 당일 재적 국회의원 수는 261명으로, 그 중에서 130명은 한나라당, 112명은 새천년민주당, 14명은 자유민주연합, 1명은 민주국민당, 4명은 무소속 의원이었다. 그 중 한나라당 의원 1명, 새천년민주당 의원 1명, 무소속 의원 1명 등 3명이 결석하여 총 258명의 의원들이 투표, sim : 38.12922668457031
    passage : [CLS] 구드룬 쉬만[SEP][CLS] 구드룬 쉬만(, 1948년 6월 9일 ~ )은 스웨덴의 정치인으로, 2005년 여성주의구상의 창당주 중 하나였으며 현재는 당 대변인이다. 일전에 1993년부터 2003년 1월까지 좌파당 대표직을 역임했으며, 2004년 탈세 스캔들 이후 좌파당을 탈당하고 여성주의 정당 창당에 집중하기 시작했다. 이후 2006년까지 국회에서 무소속으로 남아 있었다, sim : 38.101829528808594
    passage : [CLS] 민다나오산악쥐[SEP][CLS] 민다나오산악쥐 또는 흰배산악쥐("Limnomys sibuanus")는 쥐과에 속하는 설치류의 일종이다. 필리핀에서만 발견된다. 특징. 작은 설치류로 몸통 길이는 102~129mm이고 꼬리 길이는 147~174mm이다. 발 길이는 28~31mm, 귀 길이는 20~22mm, sim : 38.018280029296875
    passage : [CLS] Emerson, Lake &amp; Palmer (음반)[SEP] 사진을 찍었습니다. "새"에 나오는 대머리 이미지는 정령의 에드 캐시디와 아무런 관련이 없고 그와 닮은 것도 없습니다. 에드가 아직 정령 초상화를 가지고 있으니, 그렇다고 들었습니다."[SEP], sim : 37.94876480102539
    passage : [CLS] 구마가이 다카히로[SEP][CLS] 구마가이 다카히로(, 1995년 11월 10일 ~ )는 일본의 프로 야구 선수이며, 현재 센트럴 리그인 한신 타이거스의 소속 선수(외야수, 내야수)이다. 미야기현 센다이시 출신이다.[SEP], sim : 37.70389175415039
    -------------------------------------------
    passage : [CLS] 피어스 [UNK]거[SEP][CLS] 피어스 [UNK]거 (Piers Wenger, 1972년 6월 29일~)는 BBC 드라마 기획본부장을 지내고 있는 영국의 방송인이다. 경력. 1972년 6월 29일 영국 스태퍼드셔주의 스토크온트렌트에서 피어스 존 [UNK]거 (Piers John Wenger)라는 이름으로 태어났다. 2000년대 초부터 프로듀서로 활동, sim : 98.93701934814453
    passage : [CLS] 페드로 콘트레라스[SEP]되었다. 클럽 경력. 마드리드 출신으로 레알 마드리드 유소년부를 졸업한 콘트레라스는 1998년에 UEFA 챔피언스리그를 우승한 선수단에 포함되었지만, 그 해의 유럽대항전에 1분도 출전하지 못했다. 그는 4년 동안 2군인 레알 마드리드 카스티야에서 보내면서, 4차례 1998-99 시즌 1군 경기에 참가하였고, 소속 구단은 준우승을 거두었다. 1996-97 시즌에는 라, sim : 98.13269805908203
    passage : [CLS] Emerson, Lake &amp; Palmer (음반)[SEP]티스트는 록팝의 마이크 골드스타인과의 인터뷰에서 이를 부인했다. "위키피디아에 따르면, 그 이미지가 LA 밴드 스피릿과 어떻게든 연관되어 있다는 소문을 잠시나마 불식시키고 싶습니다. 사실은 ELP "새"를 그릴 당시 LA에서 그들에게 보낸 스피릿의 초상화도 그렸습니다. 그 그림의 구석에는 아주 비슷한 새가 등장했습니다., sim : 98.04299926757812
    passage : [CLS] 구드룬 쉬만[SEP]. 남성세를 비롯한 그의 각종 정책들은 많은 논란의 대상이 되었다.[SEP], sim : 97.93132019042969
    passage : [CLS] 주일본 네덜란드 대사관[SEP][CLS] 주일본 네덜란드 대사관(, )은 도쿄도 미나토구 시바고엔 3-6-3에 위치한 네덜란드 대사관이다. 현직 주일본 네덜란드 대사는 현재 공석 상태이다. 대사. 2023년 8월 10일자로, 시어도러스 페터스가 임시대리대사로 임명되었다. 특이 사항. 주한 네덜란드 대사관이 개설되기 직전부터 정식 주, sim : 97.781982421875
    passage : [CLS] 페드로 콘트레라스[SEP]요 바예카노에 임대되어 활약했는데, 42번의 리그 경기 중 단 1번만 결장했지만, 소속 구단은 라 리가에서 강등당했다. 콘트레라스는 1999-2000 시즌에 말라가로 이적하였다. 그 곳에서, 그는 붙박이 주전으로 활약하며 1999년부터 2003년까지 1부 리그 경기를 6번만 결장했다. 2003년 여름, 콘트레라스는 베티스로 이적, sim : 97.60859680175781
    passage : [CLS] 럭키맨[SEP][CLS] 럭키맨()은 다음을 가리킨다.[SEP], sim : 96.78964233398438
    passage : [CLS] 주일본 네덜란드 대사관[SEP]한 네덜란드 대사가 임명되기 전까지 이 대사관에 소속된 대사가 주한 대사까지 겸임한 이력이 과거에 있었다.[SEP], sim : 96.68644714355469
    passage : [CLS] 피어스 [UNK]거[SEP]하였다. 이 시기 영국의 또다른 프로듀서 빅토리아 우드와 주요 작품제작에서 10년 넘게 긴밀히 협력해 왔다. 2006년 ITV에서 방영된 &lt;주부 49세&gt;의 프로듀서를 맡아 영국 아카데미상과 RTA상을 수상하였으며, 2007년 &lt;발레 슈즈&gt;의 프로듀서를 맡았다. 2009년 BBC 웨, sim : 96.47283935546875
    passage : [CLS] 후티-사우디아라비아 분쟁[SEP][CLS] 후티-사우디아라비아 분쟁은 예멘의 후티 반군이 2015년부터 현재까지 사우디아라비아 남부 국경을 침략하여 발생한 분쟁이다. 역사. 2015년 1월, 후티 반란이 성공해, 친이란계 후티 반군이 친사우디아라비아계 예멘 수도의 대통령궁을 장악했다. 2015년 4월 2일 후티 반군이 사우디아라비아 남부, sim : 96.080810546875
    passage : [CLS] 구드룬 쉬만[SEP][CLS] 구드룬 쉬만(, 1948년 6월 9일 ~ )은 스웨덴의 정치인으로, 2005년 여성주의구상의 창당주 중 하나였으며 현재는 당 대변인이다. 일전에 1993년부터 2003년 1월까지 좌파당 대표직을 역임했으며, 2004년 탈세 스캔들 이후 좌파당을 탈당하고 여성주의 정당 창당에 집중하기 시작했다. 이후 2006년까지 국회에서 무소속으로 남아 있었다, sim : 95.08746337890625
    passage : [CLS] 대한민국 제16대 국회 후반기 의장단 선거[SEP] 등, 선거는 한나라당 측에 유리하게 진행되었다. 선거 당일 재적 국회의원 수는 261명으로, 그 중에서 130명은 한나라당, 112명은 새천년민주당, 14명은 자유민주연합, 1명은 민주국민당, 4명은 무소속 의원이었다. 그 중 한나라당 의원 1명, 새천년민주당 의원 1명, 무소속 의원 1명 등 3명이 결석하여 총 258명의 의원들이 투표, sim : 94.7779312133789
    passage : [CLS] 후티-사우디아라비아 분쟁[SEP] 국경선을 쳐들어왔다. 4년 넘게 전쟁중이다. 2016년 8월 16일, 예멘과 인접한 사우디의 나즈란 시에 후티 반군이 포격을 가해 민간인 7명이 사망했다.[SEP], sim : 94.48858642578125
    passage : [CLS] 간노의 소란[SEP] 있었다. 원래 타카우지의 아버지 사다우지는 호조 씨 소생의 적남이었던 다카요시(高[UNK])에게 가독을 남겨주고 가재인 고노 모로시게(高[UNK]重, )에게 다카요시 보좌를 명했지만 다카요시가 요절하는 바람에 우에스기 씨 소생으로서 다카요시의 이복동생으로 태어난 타카우지가 후, sim : 94.30201721191406
    passage : [CLS] 정종덕[SEP][CLS] 정종덕([UNK], 1943년 ~ 2016년 2월 16일)은 대한민국의 전 축구 감독이며, 감독으로서 카리스마 넘치는 지도력을 바탕으로 발전 가능성이 있는 선수들을 발굴하는 안목과 선수들을 조련해 성장시키는 능력에 능했던 것으로 평가되었다. 또한 건국대학교 감독 재임 시절 황선홍, 유상철, 이영표 등 여러 스타 선수들을 지도했으며, 특히 스트라이커였던 유상철을 수비형 미드필더, sim : 94.1806640625
    passage : [CLS] Emerson, Lake &amp; Palmer (음반)[SEP] 사진을 찍었습니다. "새"에 나오는 대머리 이미지는 정령의 에드 캐시디와 아무런 관련이 없고 그와 닮은 것도 없습니다. 에드가 아직 정령 초상화를 가지고 있으니, 그렇다고 들었습니다."[SEP], sim : 93.99540710449219
    passage : [CLS] 브라밤 오토모티브[SEP][CLS] 브라밤 오토모티브()는 2018년 5월에 데이비트 브라밤과 오스트레일리아의 투자자 그룹인 퓨전 캐피탈이 출범한 오스트레일리아의 자동차 제조업체로, 본사는 애들레이드에 있다.[SEP], sim : 93.94518280029297
    passage : [CLS] 얀 시만스키[SEP][CLS] 얀 시만스키()는 다음 사람을 가리킨다.[SEP], sim : 93.68018341064453
    passage : [CLS] 구마가이 다카히로[SEP][CLS] 구마가이 다카히로(, 1995년 11월 10일 ~ )는 일본의 프로 야구 선수이며, 현재 센트럴 리그인 한신 타이거스의 소속 선수(외야수, 내야수)이다. 미야기현 센다이시 출신이다.[SEP], sim : 93.63752746582031
    passage : [CLS] 정밀도와 재현율[SEP]egative Rate()와 정확도() 등이 있다.[SEP], sim : 93.5140380859375
    """
    retr_acc = retriever.val_top_k_acc()
    print(retr_acc)
    # retriever.retrieve(query="중국의 천안문 사태가 일어난 연도는?", k=20)
