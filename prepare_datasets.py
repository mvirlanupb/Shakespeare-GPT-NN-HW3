import pandas as pd
import random

if __name__=='__main__':
    random.seed(42)
    df = pd.read_csv("Shakespeare_data.csv")
    df.dropna(inplace=True)
    new_list=[]
    i=0
    while i<df.shape[0]:
        old_i=i
        play = df.iloc[i,:]["Play"]
        line_no = df.iloc[i,:]["PlayerLinenumber"]
        player=df.iloc[i,:]["Player"]
        actual_line=player+": "+df.iloc[i,:]["PlayerLine"]
        number_of_lines=1
        for j in range(i+1,df.shape[0]):
            if(df.iloc[j,:]["PlayerLinenumber"]==line_no):
                actual_line+=" "
                actual_line+=df.iloc[j,:]["PlayerLine"]
                number_of_lines+=1
            else:
                break
        new_list.append((play,line_no,player,actual_line,number_of_lines))
        i=j
        if(i==df.shape[0]-1):
            break
    new_df=pd.DataFrame(new_list,columns=["Play","Line","Player","Full Player line","Player number of lines"])
    S = 0
    threshold = int(df.shape[0]*0.85)
    for i in range(new_df.shape[0]):
        data = new_df.iloc[i,:]
        if(S<threshold):
            S+=data["Player number of lines"]
        else:
            for j in range(i,new_df.shape[0]):
                if(new_df.iloc[j,:]["Play"]!=previous_play):
                    break
            break
        previous_play=data["Play"]
    full_corpus = new_df["Full Player line"].to_list()
    train_corpus=full_corpus[:j]
    test_corpus = full_corpus[j:]
    train_file = open("train.txt","w")
    test_file=open("test.txt","w")
    full_corpus_file = open("full.txt","w")
    train_file.write("\n".join(train_corpus))
    test_file.write("\n".join(test_corpus))
    full_corpus_file.write("\n".join(full_corpus))
    train_file.close()
    test_file.close()
    full_corpus_file.close()