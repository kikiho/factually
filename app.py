import flask
import pickle
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import re
from flask import session
import random

import mythbuster

# Use pickle to load in the pre-trained model
df = pd.read_csv('data/test_comments1.csv')
vaxx_comments = df['Comment']

isVaxRelevant = pickle.load(open("model/relevance_model.sav", "rb"))
isAntiVax = pickle.load(open("model/antivax_model.sav", "rb"))
count_vec = pickle.load(open('model/count_vec.sav','rb'))

# Initialise the Flask app
app = flask.Flask(__name__, template_folder='templates')
app.secret_key = 'super secret key'

# Set up the main route
@app.route('/', methods=['GET', 'POST'])
def main():

    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        session['to_test'] = ''
        result = ""

        return(flask.render_template('main.html',
                                        comment="",
                                        result="",
                                        fact=""))

    if flask.request.method == 'POST':
        # Extract the input

        if flask.request.form['submit'] == 'Classify':
            to_test = session.get('to_test')
            pred = prediction(to_test)

            if pred == '1' or pred == 1:
                result = "This comment is anti-vax"
                fact = mythbuster.give_claim(to_test)
            if pred == '0' or pred == 0:
                result = "This comment is NOT anti-vax"
                fact = ""

            return flask.render_template('main.html',
                                         comment=to_test,
                                         result=result,
                                         fact=fact,
                                         )

        if flask.request.form['submit'] == 'Comment':
            to_test = random.choice(vaxx_comments)
            session['to_test'] = to_test
            return flask.render_template('main.html',
                                         comment=to_test,
                                         result="",
                                         fact=""
                                         )

definite_vocab = ['vaccine', 'vaccination','vax','vaxx','vaxxed','antivaccination', 'anti vaccination', 'antivax','antivaxx',
                 'injection', 'vaccines','vaccinations', 'VaccinesExposed', 'ForcedPoison' ,'FreedomExposed','unvaccinated',
                  'TheTruthAboutVaccines', 'WhatsInYourVaccines', 'antivaxxers','antivaxxer', 'vaccinated','vaxwoke',
                  'childrenshealthdefense','antivaccine','vaccinator','immunization','immunize','vaxxers','vaxxer',
                  'wakefield']

def prediction(sentence):
    prediction_related = 0
    prediction_antivax = 0
    bow = []
    bbow = []
    fbow = []

    # Clean Sentence
    sentence = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", sentence)
    sentence = re.sub(r"\,","",sentence)
    sentence = re.sub(r"\.","", sentence)
    sentence = re.sub(r"\:","",sentence)
    sentence = re.sub(r"'s", " is", sentence)
    sentence = re.sub(r"@", " ", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'", " ", sentence)
    sentence = re.sub(r"(\d+)(k)", r"\g<1>000", sentence)
    sentence = sentence.replace("\n", " ")
    sentence = re.sub(r'[^\w\s]', '', sentence).lower()

    # Transform to FBoW and BBoW
    fbow = count_vec.transform([sentence]).toarray()
    bbow = fbow
    for line in bbow:
        line = [1 for i in line if i>0]

    # Determine if Related to Vaccines
    keywords = ['vaccine', 'vaccination','vax','vaxx','vaxxed','antivaccination', 'anti vaccination', 'antivax',
                      'antivaxx','injection', 'vaccines','vaccinations', 'VaccinesExposed', 'ForcedPoison' ,'FreedomExposed',
                      'unvaccinated', 'TheTruthAboutVaccines', 'WhatsInYourVaccines', 'antivaxxers','antivaxxer',
                      'vaccinated','vaxwoke','childrenshealthdefense','antivaccine','vaccinator','immunization','immunize',
                      'vaxxers','vaxxer','wakefield','injecting']

    hashtags = ['antivax', 'declinevaccines', 'michiganforvaccinechoice', 'unvaccinated', 'stopforcedvaccinations', 'vaccinesdocauseautiam', 'bringvaxxedtorhodeisland', 'vaccinesworkkc', 'legallyavoidvaccination', 'stopblamingthegermandblamethevaccine', 'livevirusvaccine', 'vaccinescam', 'abolishvaccines', 'newvaccines', 'vaccinesarepurepoison', 'vaccineriskawareness', 'vaccinedeath', 'mandatoryvaccination', 'stopvaxforceineurope', 'nursesforvaccinesafetyalliance', 'vaxfailureupdate', 'vaxfree', 'vaccineawarness', 'vaccinesarenotsafe', 'vaxxedthemoviean', 'endvaccines', 'vaccinefraud', 'novaccinemandates', 'vaccinesandgeneticexpression', 'healthylifestylevaccine', 'vaccinesdidntsaveus', 'vaccinesarenotsafenoreffective', 'vaccinesarepoisona', 'vaccinescanada', 'childrendyingfromvaccines', 'alliwantforchristmasisvaccinechoice', 'liabilityfreevaccines', 'overvaccination', 'vaccinescauseharm', 'vaccinesexposedhistoric', 'vaxhole', 'thevaccinefreemom', 'vaccineswell', 'vaccinesharm', 'vaxxedthemovie', "globalrevolution'vaxxed", 'vaccinefreedoctor', 'vaccineskillourchildren', 'vaccineskill', 'vaccinevirustransmission', 'tellusthetruthaboutvaccines', 'swinefluvaccine', 'nomorevaccinesforme', 'fluvaccines', 'researchbeforeyouvax', 'vaccinesfail', 'staynonvax', 'illnesscomesfromvaccines', 'vaccineskillattention', 'vaccinesexposedbreaking', 'nomandatedvaccines', 'vaccineswhen', 'vaccinepoison', 'researchvaccineeffectiveness', 'vaxxedafrica', 'vaccinesafetymatters', 'vaccinesdonotwork', 'officialvaxxedpageforri', '31daysofvaccineinfo', 'vaxxedmore', 'vaccinedamageisreal', 'vaccinereactions', 'nosafevaccine', 'vaccineinjuredlivesmatter', 'stopvaccinating', 'hpvvaccinetoday', 'vaccinesexposedwant', 'petvaccineskilltoo', 'mandatoryvaccinesareunconstitutional', 'getthefactsb4uvax', 'safevaccines', 'autismvaccine', 'theautismvaccine', 'vaxxedattention', 'vaccinjneinjury', 'itsthevaccines', 'vaccinefraudvaccine', 'vaxchoice', 'vaccinesarejunkscience', 'noforcedvaccinationsunite', 'vaccinepidemic', 'ausnzvaxinjuries', 'vaccinationspreadsdisease', 'autoimmunityfromvaccination', 'vaccinesmakepeoplesick', 'researchvaccines', 'exvaxxer', '#vaxxedthemovie', 'vaxxedmichigan', 'stopforcedvaccinationwhat', 'vaxxedhouston', 'vaxxed3', 'safevaccinehoax', 'noprenatalvaccines', 'vaxxx', 'vaccinations', 'vaccinesharmi', 'teamvax', 'vaccineshedding', 'vaccinederivedpolio', 'vaccinescauseautoimmunedisease', 'coruptvaccines', 'becausevaccines', 'dtapvaccine', 'vaxxeduk', 'vaccinateornot', 'vaccin', 'vaxxedgardasil', 'vaccinesis', 'vaxxedevents', 'vaxxedaccording', 'vaxxedwhite', 'heavymetalsvaxxed', 'vaccinate', 'vaccinefree2017', 'vaxxedmom', 'vaxtruthattention', 'nonvax', 'vaxxednationtour2017', 'vaxxedpediatricians', 'vaxwithus', 'vaxopedia', 'saynotovaccines', 'profactsnotantivax', 'vaccinesafetyfirst', 'vaxxedtia', 'cdcdvaccineschedule', 'walvax2', 'ivax2protect', 'vaccinerights', 'vaccinesshed', 'maternityvaccines', 'sidsvaccines', 'needlelessfluvaccine', 'saynotomandatoryvaccines', 'hepbvaccine', 'outlawvaccines', 'endthevaccinelie', 'vaccinerisk', 'vaxxednation', 'dengvaxia', 'vaccinescauseautism', 'vaccinescantbemadesafe', 'stopmandatoryvaccinationreally', 'killervaccinesheartbreaking', 'vaccinecoercion', 'vacccinesdontwork', 'vaccinesaregarbage', 'vaccinesupdate', 'hpvvacc', 'vaccinemandate', 'thehpvvaccineispoisonattention', 'vaxxedplease', 'prochoiceisprovaccine', 'investigateb4uvaccinate', 'vaxxedthemovement', 'vaccinesa', 'antivaxx', 'vaccinesareahoax', 'vaccinesareatrick', 'mmrvaccine', 'vaccinesdontwork', 'educatednotvaccinated', 'provaccinationlogic', 'novaccinesinpregnancy', 'researchvaccineinjuryclaims', 'stopforcedvaccinationcontributions', 'investigatebeforeyouvaccinatesydney', 'wmpvaccine', 'anthraxvaccine', 'dovaccinessavelives', 'vaccinesi', 'vaccinebill', 'educatebeforeuvaccinate', 'squashthecultofvaccination', 'educatebeforeyouvaccinate', 'mmrvaccinedid', 'vaxwokean', 'vaxxedii', 'cancervaccines', 'vaxxeddoctors', 'educateb4youvaccinate', 'vaccinessavelives', 'provaxersgetblocked', 'vaccinefreedom', 'vaccinehow', 'paxvaxaerialspraying', 'vaccineeducation', 'zostavax', 'stopmandatoryvaccinationbreastmilk', 'westernaustraliansagainstmandatoryvaccination', 'vaccinedangers', 'vaccinesideeffect', 'stopverplichtvaccineren', 'vaccinesspreaddisease', 'vaccinationdoesnotequalimmunization', 'vaccinefraudrob', 'vaccinegate', 'vaccinequeenarrested', 'novax', 'stopforcedvaccinationdo', 'jeans4vaccines', 'noforcedvaccineswe', 'vaccineexemptions', 'nurseagainstmandatoryvaccines', 'ourfoodisvaxxed', 'tovaxornottovax', 'exceptwithvaccines', 'vaccinebadguys', 'overvaccinatingover', 'vaccinerisks', 'vaccineharm', 'arevaccinessafe', 'lucyturnbullprofitsfromvaccines', 'vaxinjuredlivesmatter', 'vaccinefree', 'stopforcedvaccinationmother', 'breastfeedingistheultimatevaccine', 'vaxxedcrest', 'vaccinescausedeath', 'nonvaxer', 'vaccinesuncovered', 'vaxxedthemoviewatch', 'refusevaccines', 'vaccineingredients', 'sanevax', 'vaccini', 'vaxxedstand', 'vaccinedamage', 'stopforcedvaccinationafter', 'educateb4uvaccinatesince', 'bigpharmavaccines', 'deathcomesfromvaccines', 'reseachvaccinesafety', 'nicuvaccines', 'nosafevax', 'vaxxed', 'vaccinecourtfacts', 'vaccinesdocausedementia', 'vaccinelies', 'theunvaccinatedchild', 'antivaccintaion', 'vaccinesrevealed', 'globalvaccineawareness', 'stopforcedvaccinationand', 'beunvaccinated', 'vaccinestories', 'stopcompulsoryvaccination', 'abortedfetalcellsinvaccines', 'vaccineskillpeople', 'vaccinescauseillness', 'justsaynotovaccines', 'vaccineinjurythe', 'vaccinesskipping', 'vaccinesdoharm', 'noforcedvaccines1', 'noforcedvax', 'vaccinefraudfor', 'vaccinesaredeadlybiologics', 'vaccinesdokill', 'vaccinefreebabies', 'cautionvaccines', 'novaccinationwithoutrepresentation', 'vaccinse', 'educatebeforeyouvaccine', 'missouricoalitionforvaccinechoice', 'vaxcashcow', 'noforcedvaccination', 'vaccinesit', 'worldmercuryprojectvaccines', 'bringvaxxedback', 'vaccinedamageon', 'informedvaccinationchoices', 'vaccinescanharm', 'arevaccinesafe', 'stopforcedvaccinationour', 'goodluckwithyourvaccines', 'lovemeetingunvaccinatedpeople', 'vaccineschedules', 'stopforcedvax', 'vaccinescankill', 'vaccineepidemicjust', 'vaccineinjurycompensation', 'mandatoryvaccines', 'vaxxedau', 'vaxxedwhat', 'novaccines', 'endallvaccinesthe', 'vaccinerash', 'vaccinefreenyc', 'whatsinyourvaccinesamazon', 'antivaccine', 'educatebeforevaccinate', 'vaxxedstories', 'ivax2ptotect', 'vaxharms', 'vaxxedwashington', 'stopforcedvaccinationlarry', 'keepyourvaccinatedkidshome', 'vaccineswane', 'vaccineinjuryisreal', 'vaxtruth', 'vaccinesafetycommission', 'vaccinateyourkids', 'vaxxed2', 'askmewhyidontvaccinate', 'nocompulsoryvaccinationthis', 'readavaccineinsert', 'rethinkvaccines', 'vaccinescananddocauseautistm', 'hollyslawvaccine', 'vaccinefriendlyplan', 'vaccinesjapan', 'vaccinesvitiman', 'nomorevaxlies', 'researchb4youvaccinate', 'vaccinesforprofit', 'banvaccines', 'vaxxedfortunately', 'vaxxedaustralia', 'airbournevaccination', 'vaccinesdidthis', 'vaccineskillpro', 'sidsaftervaccines', 'vaccinesdestroyfamiliesi', "vaxxeddon't", 'vaccinesurvivor', 'vaxwithwith', 'vaccineexemption', 'noobbligovaccinale', 'wearevaxxed', 'nomandatoryvaccines', 'stopmandatoryvaccinations', 'vaxed', 'whyivax', 'provaccine', 'vaccinescausebraindamage', 'ivaccinate', 'vaccinewhistleblower', 'doyouknowwhatsinavaccine', 'vaxxedisprovaccine', 'vaccinationchoice', 'edugatebeforeyouvaccinate', 'vaccineinducedseizures', 'vaccinescausemiscarriage', 'mandatoryvaccinesareadeathsentence', 'freevax', 'vaccinesarenotvegan', 'vaccineinsanity', 'stopforcedvaccinationmandatory', 'cdcvaccineschedule', 'thetruthaboutvaccines', 'stopmandatoryvaccines', 'wmpvaccines', 'nocompulsoryvaccinationplease', 'vaccinesarepoisondeath', 'researchvaccineinserts', 'eduatebeforeyouvaccinate', 'readthevaccineinsert', 'stopmandetoryvaccination', 'militaryvaccines', 'vaccinefailure', 'vaccinesexposedsneak', 'stoptheinsanityvaccines', 'selectivelyvaccinated', 'vaccinereaction', 'nationalchildhoodvaccineinjuryact', 'vaccinevictimsmemorial', 'poliovaccinecausedpolio', 'youllnevertake10000vaccinesatonce', 'vaccinesdid', 'vax', 'vaxfacts', 'antivaxxerstopglobalthreat', 'stopmandatoryvaccination', 'stopforcedvaccinationi', 'fluvax', 'hpvvaccine', 'vaccineskillrip', 'banvaccinessomething', 'vaxxedvaccines', 'nomorevaxvictims', 'vaccinefacts', 'vaccineinjuries', 'stopmandatoryvaccinationinformed', 'vaccinebraindamage', 'vaccinescauseinjury', 'vaxxedprotest', 'italyvaccines', 'nursesagainstmandatoryvaccines', 'vaccinesexposedvaccines', 'vaccineignorance', 'provaxersaresilly', 'vaccinechoice', 'vaccinelie', 'fraudvaccines', 'vaccinetruthin', 'vaccinesthank', 'unvaccinatedunite', 'vaccinesexposed', 'vaccinemisinformation', 'factsonvax', 'vaccinesmaimandkill', 'vaxxedindia', 'vaxxedbe', 'askemewhyidontvaccinate', 'vaxxedpeep', 'vaccineinjuryinterviews', 'vaccineshow', 'vaccineinternational', 'vaxxednationtour', 'ok4vaxchoice', 'vaxxedthe', 'vaccinetrauma', 'hpvvaccineontrial', 'vaccineresources', 'everyvaccineproducesharm', 'nonvaxaf', 'fighthpvvaccinetogether', 'vaxxedok', 'yesvaccinesdocauseautism', 'vaccinecompensation', 'factsonvaccineswell', 'vaccinedamagethe', 'getyourkidvaccinated', 'childrenshealthdefensedengvaxia', 'vaccineabolitionist', 'lifeaftervaccination', 'educateandnevervaccinate', 'stopforcedvaccinationmore', 'ga4vaxchoice', 'vaccinesleadingcauseofcoincidence', 'yetanothervaccinecauseddeath', 'educateb4uvaccinate', 'vaccinebigots', 'vaxxedkenya', 'vaxxedi', 'stopmandatoryvaccinationwould', 'vaccinesbreaking', 'paxvax', 'readavaccinepacketinsert', 'childrenshealthdefensevaccines', 'doctorsdonotstudyvaccinesinschool', 'vaccinationnation', 'vaccinesif', 'dangerousvaccine', 'vaxxedrevolution', 'stopforcedvaccinationyou', "vaccinesworkkcdoctor's", 'vaccinestakelives', 'stopmandatoryvaccinationthe', 'vaccineclinic', 'bigvaxxpharma', 'dtpvaccine', 'vaxxedmovement', 'bringvaxxedtotheuk', 'corruptvaccinesmemes', 'childhoodvaccineinjuryact', "stopmandatoryvaccinationi'm", 'hpvvaccineupdate', 'vaccination', "noforcedvaccinesit's", 'vaccineinjuredmothers', 'vaccineinjuriesarereal', 'vaccineskillkochani', 'vaccinevictim', 'vaccinedeathscdc', "stopforcedvaccinationit's", 'educate4thevaccineinjured', 'vaccinecourt', 'vaxxedgathr', 'vaccinespreaddisease', 'vaccinesnotsids', 'choleravaccine', 'butivaccinate', 'vaxcrap', 'vaccinemandateslearn', 'safervaccines', 'vaccineskilland', 'hpvvaccines', 'vaxxedvictimslivesmattertoo', 'vaccinescause', 'vaccinedanger', 'justsaynotomandatedvaccines', 'teamvaxxed', 'vaccinechoicewe', 'vaxwoke', 'vaccinefraudisreal', 'vaxxedworldwide', 'vaccinated', 'vaccinesdon', 'vaxxedis', 'vaxxedno', 'vaccinedeaths', 'vaccinemandates', 'vaccinationsegregation', 'rfkcommissionvaccines', 'nocompulsoryvaccination', 'vaccinescarryrisks', 'vaccinescausesids', 'vaxharm', 'vaccineinducedmeasles', 'vaccinesexposeda', 'vaccinesarentvegan', 'vaccinesinfood', 'stopforcedvaccination', 'vaccines', 'vaccineswillkillyou', 'petvaccinesarepoison', 'whatsinavaccine', 'endvaccineviolence', 'vaxxedto', 'antivaccinationleague', 'vaccinesarenotmedicine', 'vaccinescananddocauseautismannouncement', 'vaccine', 'vaxxedusalook', 'govaccinefree', 'vaccineawareness', 'whatsinyourvaccines', 'nosuchthingasgreenvaccines', 'safevax', 'vaccinationchoicethis', 'fluvaccine', 'vaccinesworkkcheads', 'vaccinesvictim', 'vaxxedinternational', 'vaccineinjuredwatch', 'vaccination50', 'provaxshill', 'vaccinekill', 'vaxrisks', 'vaxwithme', 'vaccinipuliti', 'vaccineinjurythank', 'factsonvaxregarding', 'vaxelis', 'vaccinesdocausebraindamage', 'investigatebeforeyouvaccinate', 'rethinkvaccinesto', 'vaccinelogic', 'vaxxedyou', 'microbiomevaccinesafetyproject', 'vaccineinjured', 'forcedvaccination', 'noforcedvaccines', 'unitedagainstvaccines', 'corruptvaccines', 'vaccinatedsheddisease', 'vaxtruthi', 'vaccinedeathinvestigation', 'vaxxedshare', 'stopforcedvaccinationdr', 'vaccinesdocauseautism', 'masteringvaccineinfo', 'vaccinesholistic', 'measlesvaccine', 'vaccineholocaust', 'vaccinescananddocauseautism', 'stopmandatoryvaccinationhear', 'vaccineinjury', 'vaccineswork', 'vaxxedthis', 'stopforcedvaccinationlearn', 'vaxxedflushot', 'healthdoesntcomefromvaccines', 'educatedontvaccinate', 'vaccinepapersfrom', 'provaxersbelike', 'vaxxedtruth', 'vaccineinjuryawarenessmonth', 'antivaxer', 'vaccinesaregenocide', 'vaccinesafety', 'vaccinesneversavedanyone', 'hpvvaccinedestroyslives', 'vaccinesdamagednapro', 'vaccineskilli', 'nursesagainstmandatoryvaccination', 'vaxholeof2017', 'makevaccinemanufacturersliableagain', 'vaccinecourts', 'vaccinespreadillness', 'billionspaidoutinvaccineinjurycases', 'vaccineinjuryawareness', 'vaccineabolishment', 'endvaccinesthis', 'vaxxedmovie', 'vaccinesarepoison', 'dontvaccinate', 'novaccinemandate', 'vaccineskillfull', 'educatebeforeyouvaccinatetruth', 'realnonvaxersarentprochoice', 'vaxxedwhy', 'vaxxedtexas', 'endvaccines4', 'vaxxeda', 'noforcedvaxwhile', 'vaxism', 'vaxxedfromcoveruptocatastrophe', 'contaminationvaccines', 'plotkinonvaccines', 'vaccineliberationi', 'vaccinesso', 'vaccinetruth', 'stillavaccine', 'vaxxedusa', 'endvaxinjury', 'vaccinechoicecanada', 'stopmandatoryvaccinationsenator', 'provax', 'vaxtruthflu', 'novaccinationwithoutrepresentationtorino', 'thevaccinethatkeepsonkillingin', 'vaccinesexposedthere', 'noforcedvaccinations', 'thenationalchildhoodvaccineinjuryact', 'vaxxednurses', 'vaccinesaremurder', '24dosesofvaccinesby6months', 'vaccinesareineffective', 'vaccineepidemic', 'vaccinateus', 'vaxxedworld', 'thetruthaboutvaccinesheres', 'vaccinesaretoxic', 'infanrixhexavaccine', 'thevaccinereaction', 'vaccinefraudpolio', 'vaccinedamagethis', 'vaxxedlisten', 'vaccinemisinformationdear', 'afmisvaccineinduced', 'vaccinesmercury', 'truthaboutvaccines', 'vaccinesinjure', 'vaccineagenda', 'vaccinesinjureandkill', 'vaccinecourtesy', 'vaxxedthoughts', 'worldmercuryprojectvaccine', 'vaxxedmmr', 'vaccinatie', 'contraindicatedwithvaccines', 'vaccineinjuredarmy', 'thehpvvaccineispoison', 'sciencevaccines', 'wmpvaccination', 'dtpvaccineroulette', 'forcedpoison', 'cdcwhistleblower', 'childrenshealthdefense', 'parentscalltheshots', 'thechoiceshouldbeyours', 'thehighwire', 'flyhighnicholas', 'nvic', 'drpaulthomas', 'fraudatthecdc', 'medicalchoice', 'picphysicians', 'spottingthetruthmeasles', 'spottingthetruth', 'worldmercuryproject', 'physiciansforinformedconsent', 'rfkcommission', 'injuredarmy', 'sb277', 'informedconsent', 'stopsb277', 'medicalfreedom', 'b1less', 'worldmercuryproject', 'gardasil', 'revolution4truth', 'vaccinateyourchildren']

    # Step 1: Predict based on model
    prediction_related = isVaxRelevant.predict(fbow)

    # Step 2: Double check non-related based on keywords and hashtags
    if prediction_related == '0':
        for word in sentence.split():
            if (word in hashtags or word in keywords):
                prediction_related = 1
                break

    if prediction_related == '0':
        return 0 # NOT anti-vac


    # If Vaccination-Related, predict if antivax
    prediction_antivax = isAntiVax.predict(bbow)

    return prediction_antivax[0]




if __name__ == "__main__":
    # app.config['SESSION_TYPE'] = 'filesystem'
    sess.init_app(app)

    # app.debug = True
    app.run()
