# # from datasets import list_metrics,load_metric
# # metrics_list = list_metrics()
# # len(metrics_list)
# # print(', '.join(metric for metric in metrics_list))
# # pred = "Police in the US state of California are searching for a woman who was kidnapped on Monday."
# # ref = "Denise Huskins, 30, works as a physical therapist at a Kaiser Hospital, CNN affiliate KGO reports.\nHuskins was taken from her boyfriend's residence, her cousin tells CNN affiliate KPIX."
# # metric = load_metric("rouge")
# #
# #
# #
# #
# # # preds1 = ['Singer-songwriter David Crosby has been arrested on suspicion of driving under the influence of alcohol after he hit a jogger in California.', 'People have been using Twitter to ask me what they want to know about Jesus, John the Baptist and the Shroud of Turin.', 'The Clinton Foundation has been at the centre of a fundraising controversy over the last few weeks.', 'The United States is marking the 70th anniversary of the assassination of its first ambassador to Guatemala, John Gordon Mein.', 'A British military worker has tested positive for the Ebola virus in Sierra Leone, officials say.', 'As a journalist, I seek intellectual certainty, but when it comes to faith, God, and religion, the more questions I ask the more complex the answers become.', "Nigeria's former military ruler Muhammadu Buhari has won the presidential election, early results show.", 'The Germanwings co-pilot who deliberately crashed his plane into the French Alps on Tuesday, killing all 150 people on board, had sought treatment for mental illness, reports say.', 'One of the oldest fraternities in the United States has been suspended by the University of Oklahoma.', 'In our series of letters from African-American journalists, filmmaker and columnist Farai Sevenzo looks at the growing tensions between police and communities.', 'The founder of Meerkat is answering your questions about the app, the SXSW festival and entrepreneurship.', 'A woman who was convicted of murdering her six-year-old son in 1989 has been cleared of all charges.', 'Iraqi forces are advancing on the city of Tikrit, with the help of Iran, reports say.', 'On this edition of CNN Student News, we look at some of the stories you may have missed.', 'More than 1,800 sea lion pups have been rescued so far this year, according to the US government.', "In our series of letters from women, Sargent Shriver looks at how women can help fight Alzheimer's disease."]
# # # preds2 = ['Singer-songwriter David Crosby has been arrested on suspicion of driving under the influence of alcohol after he hit a jogger in California.', 'People have been using Twitter to ask me what they want to know about Jesus, John the Baptist and the Shroud of Turin.', 'The Clinton Foundation has been at the centre of a fundraising controversy over the last few weeks.', 'The United States is marking the 70th anniversary of the assassination of its first ambassador to Guatemala, John Gordon Mein.', 'A British military worker has tested positive for the Ebola virus in Sierra Leone, officials say.', 'As a journalist, I seek intellectual certainty, but when it comes to faith, God, and religion, the more questions I ask the more complex the answers become.', "Nigeria's former military ruler Muhammadu Buhari has won the presidential election, early results show.", 'The Germanwings co-pilot who deliberately crashed his plane into the French Alps on Tuesday, killing all 150 people on board, had sought treatment for mental illness, reports say.', 'One of the oldest fraternities in the United States has been suspended by the University of Oklahoma.', 'In our series of letters from African-American journalists, filmmaker and columnist Farai Sevenzo looks at the growing tensions between police and communities.', 'The founder of Meerkat is answering your questions about the app, the SXSW festival and entrepreneurship.', 'A woman who was convicted of murdering her six-year-old son in 1989 has been cleared of all charges.', 'Iraqi forces are advancing on the city of Tikrit, with the help of Iran, reports say.', 'On this edition of CNN Student News, we look at some of the stories you may have missed.', 'More than 1,800 sea lion pups have been rescued so far this year, according to the US government.', "In our series of letters from women, Sargent Shriver looks at how women can help fight Alzheimer's disease."]
# # # labels1 = ['Accident happens in Santa Ynez, California, near where Crosby lives.\nThe jogger suffered multiple fractures; his injuries are not believed to be life-threatening.', 'Religion professor Candida Moss appears in each episode of the program.\nMoss was part of the original study to determine if relics found in Bulgaria could be the bones of John the Baptist.', 'Clinton Foundation has taken money from foreign governments.\nBill Clinton:  "I believe we have done a lot more good than harm"', 'Several U.S. diplomats have died after being attacked.\nThey include then-Ambassadors Christopher Stevens, John Mein and Francis Meloy.', 'Spokesperson: Experts are investigating how the UK military health care worker got Ebola.\nIt is being decided if the military worker infected in Sierra Leone will return to England.\nThere have been some 24,000 reported cases and 10,000 deaths in the latest Ebola outbreak.', 'Kyra Phillips became a born-again Christian as a teen.\nShe attended a Christian college, but left after her sophomore year.\nPhillips says she now considers herself a seeker of spiritual enlightenment.', "Incumbent President Goodluck Jonathan acknowledges defeat, says he delivered on promise of fair elections.\nMuhammadu Buhari's party says Jonathan called to concede even before final results are announced.\nBuhari is a 72-year-old retired major general who ruled in Nigeria in the 1980s.", 'Reuters reports German newspaper says Lubitz took break in 2009 due to depression.\nRipped medical-leave notes found at his home indicate co-pilot hid an illness, officials say.\nInvestigators found no goodbye letter or evidence of political or religious motivation.', "Sigma Alpha Epsilon is being tossed out by the University of Oklahoma.\nIt's also run afoul of officials at Yale, Stanford and Johns Hopkins in recent months.", 'Two police officers were shot Wednesday in Ferguson.\nHank Johnson, Michael Shank: Policing style needs rethink.', 'Join Meerkat founder Ben Rubin for a live chat at 2 p.m.\nET Wednesday.\nFollow @benrbn and @lauriesegallcnn on Meerkat.\nUse hashtag #CNNInstantStartups to join the conversation on Twitter.', 'Debra Milke was convicted of murder in her son\'s death, given the death penalty.\nThere was no evidence tying her to the crime, but a detective said she confessed.\nThis detective had a "history of misconduct," including lying under oath.', "Iraqi forces make some progress as they seek to advance toward Tikrit.\nThe city, best known to Westerners as Saddam Hussein's birthplace, was taken by ISIS in June.", 'This page includes the show Transcript.\nUse the Transcript to help students with reading comprehension and vocabulary.\nAt the bottom of the page, comment for a chance to be mentioned on CNN Student News.\nYou must be a teacher or a student age 13 or older to request a mention on the CNN Student News Roll Call.', '"There has been an unusually high number of sea lions stranded since January," NOAA representative says.\nThe speculation is mothers are having difficulty finding food, leaving pups alone too long or malnourished.', "Maria Shriver's father was stricken by Alzheimer's, a growing scourge in U.S.\nWomen are disproportionately affected as sufferers and caregivers, she says.\nWipe Out Alzheimer's Challenge is launching to fill in for lagging government funding, she says."]
# # # labels2 = ['Accident happens in Santa Ynez, California, near where Crosby lives.\nThe jogger suffered multiple fractures; his injuries are not believed to be life-threatening.', 'Religion professor Candida Moss appears in each episode of the program.\nMoss was part of the original study to determine if relics found in Bulgaria could be the bones of John the Baptist.', 'Clinton Foundation has taken money from foreign governments.\nBill Clinton:  "I believe we have done a lot more good than harm"', 'Several U.S. diplomats have died after being attacked.\nThey include then-Ambassadors Christopher Stevens, John Mein and Francis Meloy.', 'Spokesperson: Experts are investigating how the UK military health care worker got Ebola.\nIt is being decided if the military worker infected in Sierra Leone will return to England.\nThere have been some 24,000 reported cases and 10,000 deaths in the latest Ebola outbreak.', 'Kyra Phillips became a born-again Christian as a teen.\nShe attended a Christian college, but left after her sophomore year.\nPhillips says she now considers herself a seeker of spiritual enlightenment.', "Incumbent President Goodluck Jonathan acknowledges defeat, says he delivered on promise of fair elections.\nMuhammadu Buhari's party says Jonathan called to concede even before final results are announced.\nBuhari is a 72-year-old retired major general who ruled in Nigeria in the 1980s.", 'Reuters reports German newspaper says Lubitz took break in 2009 due to depression.\nRipped medical-leave notes found at his home indicate co-pilot hid an illness, officials say.\nInvestigators found no goodbye letter or evidence of political or religious motivation.', "Sigma Alpha Epsilon is being tossed out by the University of Oklahoma.\nIt's also run afoul of officials at Yale, Stanford and Johns Hopkins in recent months.", 'Two police officers were shot Wednesday in Ferguson.\nHank Johnson, Michael Shank: Policing style needs rethink.', 'Join Meerkat founder Ben Rubin for a live chat at 2 p.m.\nET Wednesday.\nFollow @benrbn and @lauriesegallcnn on Meerkat.\nUse hashtag #CNNInstantStartups to join the conversation on Twitter.', 'Debra Milke was convicted of murder in her son\'s death, given the death penalty.\nThere was no evidence tying her to the crime, but a detective said she confessed.\nThis detective had a "history of misconduct," including lying under oath.', "Iraqi forces make some progress as they seek to advance toward Tikrit.\nThe city, best known to Westerners as Saddam Hussein's birthplace, was taken by ISIS in June.", 'This page includes the show Transcript.\nUse the Transcript to help students with reading comprehension and vocabulary.\nAt the bottom of the page, comment for a chance to be mentioned on CNN Student News.\nYou must be a teacher or a student age 13 or older to request a mention on the CNN Student News Roll Call.', '"There has been an unusually high number of sea lions stranded since January," NOAA representative says.\nThe speculation is mothers are having difficulty finding food, leaving pups alone too long or malnourished.', "Maria Shriver's father was stricken by Alzheimer's, a growing scourge in U.S.\nWomen are disproportionately affected as sufferers and caregivers, she says.\nWipe Out Alzheimer's Challenge is launching to fill in for lagging government funding, she says."]
# # # metric.add_batch(predictions = preds1, references = labels1)
# # # result = metric.compute()
# # # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
# # # result = {k: round(v, 4) for k, v in result.items()}
# # # print(result)
# # # metric.add_batch(predictions = preds2, references = labels2)
# # #
# # # result = metric.compute()
# # # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
# # # result = {k: round(v, 4) for k, v in result.items()}
# # # print(result)
# from transformers import AutoTokenizer, AutoConfig
# 
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
# tokenizer.save_pretrained('./test/saved_model/')
# # AutoTokenizer.from_pretrained("./token")
# tokenizer = AutoTokenizer.from_pretrained('./test/bert_saved_model/')
# # config = AutoConfig.from_pretrained("facebook/bart-base")
# # # print(config)
# # config.save_pretrained("bart-base.json")
# from datasets import load_dataset
import nltk
from nltk import ngrams

sentence = 'this is a foo bar sentences and i want to ngramize it'

n = 6
sixgrams = ngrams(sentence.split(), n)

print(('this1', 'is', 'a', 'foo', 'bar', 'sentences') in list(sixgrams))

print(nltk.tokenize.word_tokenize(sentence))
# sents = nltk.sent_tokenize("And now for something completely different. I love you.")
# word = []
# for sent in sents:
#     word.append(nltk.word_tokenize(sent))
# print(word)
