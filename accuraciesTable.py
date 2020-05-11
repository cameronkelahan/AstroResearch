import plotly.graph_objects as go

headerColor = 'grey'
rowEvenColor = 'lightgrey'
rowOddColor = 'white'

# ['<b>Model</b>','<b>Training Accuracy</b>','<b>Training F1 Score</b>','<b>Test Accuracy</b>',
#             '<b>Test F1 Score</b>','<b>KNN K-Value</b>', '<b># of Nodes Per Layer</b>', '<b>Activation of Layers</b>'],

#[
#       ['<b>KNN Unweighted</b>', '<b>KNN Weighted</b>', '<b>Keras NN 3 Layers</b>', '<b>Keras NN 4 Layers</b>'],
#       ['96.3%', 'N/A', '71.6%', '71.6%'],
#       [.960, 'N/A', .723, .736],
#       ['82.1%', '82.1%', '67.9%', '75.0%'],
#       [.839, .839, .640, .741],
#       [10, 10, 'N/A', 'N/A'],
#       ['N/A', 'N/A', '<b>1)</b> 12                          <b>2)</b> 8                             <b>3)</b> 1',
#        '''<b>1)</b> 8                             <b>2)</b> 12                            <b>3)</b> 4
#                             <b>4)</b> 1'''],
#       ['N/A', 'N/A', '''<b>1)</b> ReLU                        <b>2)</b> ReLU                       <b>3)</b> Sigmoid''',
#        '''<b>1)</b> ReLU                        <b>2)</b> ReLU                        <b>3)</b> ReLU
#                       <b>4)</b> Sigmoid''']],

fig = go.Figure(data=[go.Table(
  header=dict(
    values=['<b></b>','<b>KNNUnweighted</b>','<b>KNN Weighted</b>','<b>Keras NN 3 Layers</b>',
            '<b>Keras NN 4 Layers</b>'],
    line_color='darkslategray',
    fill_color=headerColor,
    align=['left','center'],
    font=dict(color='white', size=12)
  ),
  cells=dict(
    values=[
      ['<b>Training Accuracy</b>', '<b>Training F1 Score</b>', '<b>Test Accuracy</b>', '<b>Test F1 Score</b>',
       '<b>KNN K-Value</b>', '<b># of Nodes Per Layer</b>', '<b>Activation of Layers</b>'],
      ['96.3%', .960, '82.1%', .839, 10, 'N/A', 'N/A'],
      ['N/A', 'N/A', '82/1%', .839, 10, 'N/A', 'N/A'],
      ['71.6%', .723, '67.9%', .640, 'N/A', '''<b>1)</b> 12                                                <b>2)</b> 8                                             
      <b>3)</b> 1''', '''<b>1)</b> ReLU                                              <b>2)</b> ReLU                       
                      <b>3)</b> Sigmoid'''],
      ['71.6%', .736, '75.0%', .741, 'N/A', '''<b>1)</b> 8                                                  <b>2)</b> 12                                            
      <b>3)</b> 4                                                    <b>4)</b> 1''',
       '''<b>1)</b> ReLU                                              <b>2)</b> ReLU                                      
       <b>3)</b> ReLU                                               <b>4)</b> Sigmoid''']],
    line_color='darkslategray',
    # 2-D list of colors for alternating rows
    fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor, rowEvenColor, rowOddColor,
                   rowEvenColor]*5],
    align = ['left', 'center'],
    font = dict(color = 'darkslategray', size = 11)
    ))
])

fig.show()