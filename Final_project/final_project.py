import numpy as np
import pickle
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
data_file='data/web.pkl'
with open(data_file, 'rb') as f:
    data = pickle.load(f)

bow_train = data['train_bow'].T
label_train = data['train_label']
voc = data['voc']
bow_test = data['test_bow'].T
label_test  = data['test_label']
for K in [4,8,10,16,20,100,200,400,1000]:
    lda = LatentDirichletAllocation(n_components=K,learning_method='batch',verbose=True,max_iter=100)
    lda.fit(bow_train)
    with open('ldamodel_batch'+str(K)+'.pkl','wb') as f:
        pickle.dump(lda,f)
ppltrain=[]
ppltest=[]
for K in [4,8,10,16,20,100,200,400]:
    with open('ldamodel_batch{}.pkl'.format(K),'rb') as f:
         lda = pickle.load(f)
    ppltrain.append(lda.perplexity(bow_train))
    ppltest.append(lda.perplexity(bow_test))
plt.plot([4,8,10,16,20,100,200,400],ppltrain,label='Train',marker='o')
plt.plot([4,8,10,16,20,100,200,400],ppltest,label='Test',marker='d',ls='--')
plt.grid(True)
plt.legend()
plt.xlabel('k')
plt.ylabel('Perplexity')
plt.show()

for K in [4,8,10,16,20,100,200,400,1000]:
    with open('ldamodel_batch{}.pkl'.format(K),'rb') as f:
         lda = pickle.load(f)
    ppltrain.append(lda.perplexity(bow_train))
    ppltest.append(lda.perplexity(bow_test))
plt.plot([4,8,10,16,20,100,200,400,1000],ppltrain,label='Train',marker='o')
plt.plot([4,8,10,16,20,100,200,400,1000],ppltest,label='Test',marker='d',ls='--')
plt.grid(True)
plt.legend()
plt.xlabel('k')
plt.ylabel('Log-Likelihood')
plt.show()



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
ppltrain=[]
ppltest=[]

clf = LogisticRegression(random_state=0).fit(bow_train, label_train)
y_pred = clf.predict(bow_test)
acc = accuracy_score(y_pred,label_test)
macprecision,macrecall,macf1,_ = precision_recall_fscore_support(y_pred,label_test,average='macro')
micprecision,micrecall,micf1,_ = precision_recall_fscore_support(y_pred,label_test,average='micro')
accs=[]
macps=[]
micps=[]
macrs=[]
micrs=[]
macf1s=[]
micf1s=[]
for K in [4,8,10,16,20,100,200,400,1000]:
    with open('ldamodel_batch{}.pkl'.format(K),'rb') as f:
         lda = pickle.load(f)
    clf = LogisticRegression(random_state=0).fit(lda.transform(bow_train), label_train)
    y_pred = clf.predict(lda.transform(bow_test))
    acc_l = accuracy_score(y_pred, label_test)
    macprecision_l, macrecall_l, macf1_l, _ = precision_recall_fscore_support(y_pred, label_test, average='macro')
    micprecision_l, micrecall_l, micf1_l, _ = precision_recall_fscore_support(y_pred, label_test, average='micro')
    accs.append(acc_l)
    macps.append(macprecision_l)
    micps.append(micprecision_l)
    macrs.append(macrecall_l)
    micrs.append(micrecall_l)
    macf1s.append(macf1_l)
    micf1s.append(micf1_l)
plt.axhline(acc,label='Original data',marker='o')
plt.plot([4,8,10,16,20,100,200,400,1000],accs,label='Compressed data (k)',marker='d',c='tab:red',ls='--')
plt.grid(True)
plt.legend()
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
plt.close()

plt.axhline(acc,label='Original data',marker='o')
plt.plot([4,8,10,16,20,100,200,400,1000],macps,label='Compressed data (k)',marker='d',c='tab:red',ls='--')
plt.grid(True)
plt.legend()
plt.xlabel('k')
plt.ylabel('Macro Precision')
plt.tight_layout()
plt.show()
plt.close()

plt.axhline(acc,label='Original data',marker='o')
plt.plot([4,8,10,16,20,100,200,400,1000],micps,label='Compressed data (k)',marker='d',c='tab:red',ls='--')
plt.grid(True)
plt.legend()
plt.xlabel('k')
plt.ylabel('Micro Precision')
plt.tight_layout()
plt.show()
plt.close()

plt.axhline(acc,label='Original data',marker='o')
plt.plot([4,8,10,16,20,100,200,400,1000],macrs,label='Compressed data (k)',marker='d',c='tab:red',ls='--')
plt.grid(True)
plt.legend()
plt.xlabel('k')
plt.ylabel('Macro Recall')
plt.tight_layout()
plt.show()
plt.close()

plt.axhline(acc,label='Original data',marker='o')
plt.plot([4,8,10,16,20,100,200,400,1000],micrs,label='Compressed data (k)',marker='d',c='tab:red',ls='--')
plt.grid(True)
plt.legend()
plt.xlabel('k')
plt.ylabel('Micro Recall')
plt.tight_layout()
plt.show()
plt.close()


plt.axhline(acc,label='Original data',marker='o')
plt.plot([4,8,10,16,20,100,200,400,1000],macf1s,label='Compressed data (k)',marker='d',c='tab:red',ls='--')
plt.grid(True)
plt.legend()
plt.xlabel('k')
plt.ylabel('Macro F1')
plt.tight_layout()
plt.show()
plt.close()

plt.axhline(acc,label='Original data',marker='o')
plt.plot([4,8,10,16,20,100,200,400,1000],micf1s,label='Compressed data (k)',marker='d',c='tab:red',ls='--')
plt.grid(True)
plt.legend()
plt.xlabel('k')
plt.ylabel('Micro F1')
plt.tight_layout()
plt.show()
plt.close()


from wordcloud import WordCloud

with open('ldamodel_batch{}.pkl'.format(8), 'rb') as f:
    lda = pickle.load(f)
for i in range(8):
    topic = lda.components_[i]
    text=dict(zip(voc, topic))
    wc = WordCloud(background_color='white', width = 500, height=500)
    result=wc.fit_words(text)
    plt.imshow(result, interpolation='bilinear')
    plt.axis('off')
    plt.title('Topic {}'.format(i+1))
    plt.tight_layout()
    plt.savefig('topic{}.pdf'.format(i+1))
    # plt.show()
    plt.close()
    plt.clf()
