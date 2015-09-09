import plotly.plotly as py
from plotly.graph_objs import *

def readdata(file, topicCount):
  dataset = {}

  reader = open(file, 'r')
  reader.seek(0)

  count = 0
  topic = -1 
  while True:
    originalline = reader.readline()
    line = originalline.split('\n')

    if 'Topic' in line[0]:
      count += 1
      if count > topicCount:
        break

      topic = int(line[0].split(':')[1])
      dataset[topic] = []
    else:
      dataset[topic].append(line[0])

  reader.close()
  return dataset


def normalize(values):
  total = 0.0
  for value in values:
    vals = value.split(':')
    total += float(vals[1])

  return [total*1000/len(values)]


def build_text(key, values):
  result = 'Topic ' + str(key) + ':<br>' 

  for value in values:
    result += (value + '<br>')

  return [result]


def build_trace(dataset):
  traces = []
  for key in sorted(dataset.keys()):
    trace = 'trace' + str(key)
    trace = Scatter(
      x = [key],
      y = normalize(dataset[key]),
      text = build_text(key, dataset[key]),
      mode = 'markers',
      name = key,
      marker = Marker(
        sizemode = 'diameter',
        sizeref = 0.85,
        size=[normalize(dataset[key])],
        line=Line(
            width=2
        ),
      )
    )

    traces.append(trace)

  return traces

def build_layout():
  layout = Layout(
    title='Cluster of Topic0 - Topic10 by LDP',
    showlegend=False,
    height = 600,
    width = 800,
    xaxis=XAxis(
        title='Topic ID',
        gridcolor='rgb(255, 255, 255)',
        zerolinewidth=1,
        ticklen=5,
        gridwidth=1,
    ),
    yaxis=YAxis(
        title='Group Frequency (1000x)',
        gridcolor='rgb(255, 255, 255)',
        zerolinewidth=1,
        ticklen=5,
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
  )
  return layout

def display(dataset):
  for k in dataset:
    print '\nkey: ', k
    print '\nvalues: ', dataset[k]

def main(file):
  dataset = readdata(file, 10)
  traces = build_trace(dataset)
  print traces
  layout = build_layout()
  fig = Figure(data=Data(traces), layout=layout)
  plot_url = py.plot(fig, filename='task1.1 LDP Topic Sample')

if __name__ == '__main__':
  main('/home/yfliu/Documents/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/sample_topics.txt')
