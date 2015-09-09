import json
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import models
from gensim import matutils
from nltk.tokenize import sent_tokenize
import plotly.plotly as py
from plotly.graph_objs import *

base_path = '/home/yfliu/Documents/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task1.2/'

def buildBusinessInfo(businessFile, target_restaurant):
    businessInfo = {}
    target_rastaurantID = []

    reader = open(businessFile, 'r')
    reader.seek(0)

    line = reader.readline()
    while line:
        business_json = json.loads(line)
        businessID = business_json['business_id']

        if businessID not in businessInfo:
            businessInfo[businessID] = {}

        name = business_json['name'].replace('/','&')
        if name in target_restaurant:
            target_rastaurantID.append(businessID)
            target_restaurant.remove(name)

        businessInfo[businessID]['name'] = name 
        businessInfo[businessID]['stars'] = business_json['stars'] 
        businessInfo[businessID]['reviews'] = {} 

        line = reader.readline()

    reader.close()
    print 'build business info finished'
    return businessInfo, target_rastaurantID


def buildBusinessReviewMapping(reviewFile, businessInfo):

    reader = open(reviewFile, 'r')
    reader.seek(0)
    line = reader.readline()

    count = 0
    while line:
        review_json = json.loads(line)
        businessID = review_json['business_id']

        if businessID not in businessInfo:
            continue

        count += 1
        if count > 500000:
            break

        stars = int(review_json['stars'])
        review = review_json['text']

        if stars not in businessInfo[businessID]['reviews']:
            businessInfo[businessID]['reviews'][stars] = []

        for rev in review.split('\n\n'):
            businessInfo[businessID]['reviews'][stars].append(rev)

        line = reader.readline()

    reader.close()
    print 'build business review mapping finished'
    return businessInfo 


def generateTopicModel(businessReview):
    for businessID in businessReview.keys():
        restaurant = businessReview[businessID]['name']
        reviews = businessReview[businessID]['reviews']

        for star in reviews.keys():
            sample_file = base_path + 'data/input_restaurant_' + restaurant + '_' + str(star) + '.txt'
            output_file = base_path + 'data/output_restaurant_' + restaurant + '_' + str(star) + '.txt'

            with open(sample_file, 'w') as writer:
                for rev in reviews[star]:
                    try: 
                        writer.write(rev + '\n')
                    except:
                        continue

            print 'lda for: ' + restaurant
            try:
                lda(5, 10, sample_file, 10, output_file)
            except:
                continue


def generateGraph(businessInfo, target_restaurantID):
    import plotly.plotly as py
    from plotly.graph_objs import *

    print businessInfo[target_restaurantID[0]]['name']
    total1 = 0.0
    for i in range(1,6):
        total1 += len(businessInfo[target_restaurantID[0]]['reviews'][i])

    print total1

    print businessInfo[target_restaurantID[1]]['name']
    total2 = 0.0
    for i in range(1,6):
        total2 += len(businessInfo[target_restaurantID[1]]['reviews'][i])

    print total2

    print businessInfo[target_restaurantID[2]]['name']
    total3 = 0.0
    for i in range(1,6):
        total3 += len(businessInfo[target_restaurantID[2]]['reviews'][i])

    print total3

    trace1 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[0]]['reviews'][5])/total1)],
        y=[businessInfo[target_restaurantID[0]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(38, 24, 74, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace2 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[1]]['reviews'][5])/total2)],
        y=[businessInfo[target_restaurantID[1]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(38, 24, 74, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace3 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[2]]['reviews'][5])/total3)],
        y=[businessInfo[target_restaurantID[2]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(38, 24, 74, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace4 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[0]]['reviews'][4])/total1)],
        y=[businessInfo[target_restaurantID[0]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(38, 24, 74, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace5 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[1]]['reviews'][4])/total2)],
        y=[businessInfo[target_restaurantID[1]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(71, 58, 131, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace6 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[2]]['reviews'][4])/total3)],
        y=[businessInfo[target_restaurantID[2]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(71, 58, 131, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace7 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[0]]['reviews'][3])/total1)],
        y=[businessInfo[target_restaurantID[0]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(71, 58, 131, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace8 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[1]]['reviews'][3])/total2)],
        y=[businessInfo[target_restaurantID[1]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(71, 58, 131, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace9 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[2]]['reviews'][3])/total3)],
        y=[businessInfo[target_restaurantID[2]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(122, 120, 168, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace10 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[0]]['reviews'][2])/total1)],
        y=[businessInfo[target_restaurantID[0]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(122, 120, 168, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace11 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[1]]['reviews'][2])/total2)],
        y=[businessInfo[target_restaurantID[1]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(122, 120, 168, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace12 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[2]]['reviews'][2])/total3)],
        y=[businessInfo[target_restaurantID[2]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(122, 120, 168, 0.8)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace13 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[0]]['reviews'][1])/total1)],
        y=[businessInfo[target_restaurantID[0]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(164, 163, 204, 0.85)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace14 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[1]]['reviews'][1])/total2)],
        y=[businessInfo[target_restaurantID[1]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(164, 163, 204, 0.85)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    trace15 = Bar(
        x=[int(100*len(businessInfo[target_restaurantID[2]]['reviews'][1])/total3)],
        y=[businessInfo[target_restaurantID[2]]['name']],
        orientation='h',
        marker=Marker(
            color='rgba(164, 163, 204, 0.85)',
            line=Line(
                color='rgb(248, 248, 249)',
                width=1
            )
        )
    )
    data = Data([trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9, trace10, trace11, trace12, trace13, trace14, trace15])
    layout = Layout(
        showlegend=False,
        width=800,
        height=600,
        xaxis=XAxis(
            range=[0, 100.00],
            domain=[0.15, 1],
            type='linear',
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False
        ),
        yaxis=YAxis(
            range=[-0.5, 3.5],
            type='category',
            autorange=True,
            showgrid=False,
            zeroline=False,
            showline=False,
            showticklabels=False
        ),
        annotations=Annotations([
            Annotation(
                x=0.14,
                y=0,
                xref='paper',
                yref='y',
                text=businessInfo[target_restaurantID[0]]['name'],
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(67, 67, 67)'
                ),
                xanchor='right',
                align='right'
            ),
            Annotation(
                x=2,
                y=0,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[0]]['reviews'][5])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=23,
                y=0,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[0]]['reviews'][4])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=53,
                y=0,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[0]]['reviews'][3])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=71,
                y=0,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[0]]['reviews'][2])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=88,
                y=0,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[0]]['reviews'][1])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=0.14,
                y=1,
                xref='paper',
                yref='y',
                text=businessInfo[target_restaurantID[1]]['name'],
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(67, 67, 67)'
                ),
                xanchor='right',
                align='right'
            ),
            Annotation(
                x=15,
                y=1,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[1]]['reviews'][5])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=55,
                y=1,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[1]]['reviews'][4])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=82,
                y=1,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[1]]['reviews'][3])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=90,
                y=1,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[1]]['reviews'][2])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=95,
                y=1,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[1]]['reviews'][1])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=0.14,
                y=2,
                xref='paper',
                yref='y',
                text=businessInfo[target_restaurantID[2]]['name'],
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(67, 67, 67)'
                ),
                xanchor='right',
                align='right'
            ),
            Annotation(
                x=5,
                y=2,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[2]]['reviews'][5])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=22,
                y=2,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[2]]['reviews'][4])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=55,
                y=2,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[2]]['reviews'][3])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=77,
                y=2,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[2]]['reviews'][2])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=90,
                y=2,
                xref='x',
                yref='y',
                text=str(len(businessInfo[target_restaurantID[2]]['reviews'][2])),
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(248, 248, 255)'
                )
            ),
            Annotation(
                x=4,
                y=1.1,
                xref='x',
                yref='paper',
                text='Rating<br>5',
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(67, 67, 67)'
                )
            ),
            Annotation(
                x=23,
                y=1.1,
                xref='x',
                yref='paper',
                text='Rating<br>4',
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(67, 67, 67)'
                )
            ),
            Annotation(
                x=53,
                y=1.1,
                xref='x',
                yref='paper',
                text='Rating<br>3',
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(67, 67, 67)'
                )
            ),
            Annotation(
                x=76,
                y=1.1,
                xref='x',
                yref='paper',
                text='Rating<br>2',
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(67, 67, 67)'
                )
            ),
            Annotation(
                x=90,
                y=1.1,
                xref='x',
                yref='paper',
                text='Rating<br>1',
                showarrow=False,
                font=Font(
                    family='Arial',
                    size=14,
                    color='rgb(67, 67, 67)'
                ),
                arrowhead=0,
                arrowsize=1.1,
                ax=10,
                ay=-40
            )
        ]),
        margin=Margin(
            l=120,
            r=10,
            b=80,
            t=140
        ),
        paper_bgcolor='rgb(248, 248, 255)',
        plot_bgcolor='rgb(248, 248, 255)',
        barmode='stack'
    )
    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig)
    

def lda(K, numfeatures, sample_file, num_display_words, outputfile):
    K_clusters = K
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=numfeatures,
                                     min_df=2, stop_words='english',
                                     use_idf=True)

    text = []
    with open (sample_file, 'r') as f:
        text = f.readlines()

    t0 = time()
    #print("Extracting features from the training dataset using a sparse vectorizer")
    X = vectorizer.fit_transform(text)
    #print("done in %fs" % (time() - t0))
    #print("n_samples: %d, n_features: %d" % X.shape)
    
    # mapping from feature id to acutal word
    id2words ={}
    for i,word in enumerate(vectorizer.get_feature_names()):
        id2words[i] = word

    t0 = time()
    #print("Applying topic modeling, using LDA")
    #print(str(K_clusters) + " topics")
    corpus = matutils.Sparse2Corpus(X,  documents_columns=False)
    lda = models.ldamodel.LdaModel(corpus, num_topics=K_clusters, id2word=id2words)
    #print("done in %fs" % (time() - t0))
        
    output_text = []
    for i, item in enumerate(lda.show_topics(num_topics=K_clusters, num_words=num_display_words, formatted=False)):
        output_text.append("Topic: " + str(i))
        for weight,term in item:
            output_text.append( term + " : " + str(weight) )

    #print "writing topics to file:", outputfile
    with open ( outputfile, 'w' ) as f:
        f.write('\n'.join(output_text))
        

def main():
    target_restaurant = ['Jersey Mike\'s Subs','24 Seven Cafe','Ah-So Sushi & Steak']
    businessInfo, target_restaurantID = buildBusinessInfo(base_path + 'yelp_academic_dataset_business.json', target_restaurant)
    businessReview = buildBusinessReviewMapping(base_path + 'yelp_academic_dataset_review.json', businessInfo)
    #$generateTopicModel(businessReview)
    generateGraph(businessReview, target_restaurantID)


if __name__ == '__main__':
    main()