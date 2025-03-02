import json
from pydantic import BaseModel
from tqdm import tqdm

data = ['After its failure to overthrow Assad in Syria, the US has shifted its focus toward Venezuela, a country that Washington is now trying to destabilise under the pretext of “restoring democracy”. In reality, the US, which is an “agent of the New World Order”, is seeking to gain control over Venezuela’s oil resources by creating a crisis in the country followed by US military intervention.\n\nThe US destabilisation of Venezuela and its efforts to topple Maduro follow a familiar pattern of US interventions in Syria, Iraq, Libya and Ukraine, also justified as necessary for “promoting democracy”.',
 "Data from a Russian satellite revealed a strong Electro-magnetic radiation pulse was fired on Michael Jackson's whereabouts immediately before his death. CIA needed to get rid of him because he was to warn the world before an immediate mass genocide of Palestinians through his \nsong We've had enough.",
 'Today I Rise: This Beautiful Short Film Is Like a Love Poem For Your Heart and Soul Nov 10, 2016 2 0 \nThis beautiful short film will have you warm, fuzzy, and inspired. Here’s “Today I Rise”. \n“The world is missing what I am ready to give: My Wisdom, My Sweetness, My Love and My hunger for Peace.” \n“Where are you? Where are you, little girl with broken wings but full of hope? Where are you, wise women covered in wounds? Where are you?” \nh/t Films for Action , Image: Banksy Vote Up',
 'Russians in America: Seeking success in Silicon Valley / All news / Russians in America: Seeking success in Silicon Valley November 23, 2016 \nIn the last few years, many Russian developers and startups have moved to Silicon Valley in the hopes of conquering global markets. But only a select few have had any success. \nDespite the illusion many Russians have that in such a place, in such a productive ecosystem, with so much talent, it will be easier to gain success than in their own country, the reality is much more complicated. Russian entrepreneurs and investors face a particular set of challenges breaking into the market, and perhaps not the ones they would expect. \n“One of the difficulties for Russia’s post-Soviet startups is the culture shock, when they try to use techniques from the old country in new environments and expect the same results,” says Igor Shoifot, founder of the “Happy Farm” business incubator. Read More Related Posts',
 'Hillary Clinton When They Asked Her What She Thinks of Hillary Clinton, They Never Expected Her to Say THIS! \n0 comments \nKids say the darndest things… ADORABLE! "My dad told me that Hillary Clinton LIES A LOT, so if she wins she might take over the country! " @realDonaldTrump #VoteTrump pic.twitter.com/cHrP8lkPbS',
 'Outspoken conservative Tomi Lahren makes some great points about the  failing  mainstream media and the  failing  Democrat party blaming  fake news conservative news for Hillary s loss. To say Facebook favoritism got Donald Trump elected is the most ridiculous thing I ve heard yet. We already know Facebook is in the business of censoring and de-prioritizing conservative leading posts.',
 "Teens walk free after gang-rape conviction Judge said group who left girl, 14, for dead appeared 'repentant' Published: 20 mins ago \n(Deutsche Welle) In the wake of the news that a group of teenagers were unlikely to see any real punishment for gang-raping a 14-year-old girl and leaving her for dead, citizens of the German city of Hamburg called for new rules regarding violent crime committed by minors. On Monday, an online petition calling for the teens to see jail time had garnered some 21,000 signatures. \n“The sexual self-determination and integrity of a woman must have more weight than any concern for the perpetrators,” [of sexual crimes,] says the petition. \nAccording to an update on the Change.org petition, state prosecutors in Hamburg have said they will explore a way to make sure that the teens are punished despite laws that make it difficult for minors to be prosecuted and sentenced to detention.",
 'The Minister of Infrastructure of Ukraine Vladislav Krikliy said in an interview with Radio Liberty that Kyiv decided to restore passenger transport links with the Crimea.',
 'The foreign policy of Estonia is controlled by Washington and the guidelines for its socio-economic development come mainly from Brussels.',
 'This is a great flashback of all those who claim to have a crystal ball on who the nominee for the RNC will be Donald Trump will not be the nominee',
 'The European Union does not care what will happen with the border between Northern Ireland and the Republic of Ireland. For European bureaucrats there is no difference how the United Kingdom leaves the European Union, it simply must leave.',
 'President Trump took Secretary Kelly (DHS), Secretary Ross (Commerce), Secretary Mnuchin (Treasury), Secretary Shulkin (VA), together with their spouses to Trump National Golf Club in Virginia.Accompanying the cabinet was key staff: Press Secretary Sean Spicer, Senior Adviser Steve Bannon, and Chief of Staff Reince Priebus.',
 'Imagine losing one of your senses: sight, touch, taste, etc… It’d be quite difficult, yes? Now, imagine losing the ability to use your senses, despite still retaining them. It might be agreed that...',
 "According to a poll conducted by the European Commission, 55% of the EU population think that the process of integration of the majority of migrants in the country is unsuccessful.\n\nFor example, 73% of Sweden's population believes that integration of migrants is unsuccessful, in France - 64% and in Germany - 63%.",
 'At present, NATO does not want to accept Ukraine and, probably, will never accept it. For the West, Ukraine is nothing more than a buffer zone, which separates Europe from its geopolitical rival – Russia – and its role is to put pressure on it.',
 'ESPN host Jemele Hill called President Trump and Kid Rock  white supremacists . It didn t end there she said in one of several tweets that Trump is  unfit  to be president. The screenshot below gives you an idea of how Hill represents ESPN on twitter. It s disgusting!What did ESPN do? You guessed it NOTHING! They tweeted out a mild apology and that s it!This is after Mike Ditka was fired for supporting President Trump and criticizing Obama.  Kurt Schilling was also fired after posting a picture on social media against transgender bathrooms.Why hasn t this woman been fired by ESPN?Please contact ESPN to let them know how you feel about this outrage: ESPNTucker Carlson spoke with the awesome Clay Travis last night. Travis shed some light on why Hill is still employed by ESPN:',
 'Print \nFor those who were hoping or dreading the continued presence of Crazy Joe Biden in a presidential administration after the Barack Obamas belatedly leave the White House, the loose-lipped 73-year-old has confirmed that the nation won’t have Joe Biden to kick around any more. \nBiden, who was rumored to be on Hillary Clinton’s short list for secretary of state and prior to that was being groomed as a late contender for the Democratic Party nomination in the event Clinton failed to carry California in the primary, told a Minnesota radio station Friday he has “no intention” of serving in Hillary Clinton’s administration as secretary of state or in any other role. \n“I don’t want to remain in the administration.… I have no intention of staying involved,” Biden told Minnesota radio station KBJR . \nHe added, though, that he wanted to “help [Clinton] if I can in any way I can.” \nIt’s not clear whether Biden’s lack of interest is absolute, or if he could be talked into serving as secretary of state despite his doubts. \nThis report, by Blake Neff, was cross-posted by arrangement with the Daily Caller News Foundation.',
 'The European Union has Nazi Roots. The process of EU integration, from the outset, was coordinated by the CIA to create an anti-Russian geopolitical bloc in Europe. But the CIA did not build the European Union from scratch. The most important contributions were made earlier by the Nazis.',
 'NATO intends to turn the entire country of Montenegro into its military base by "occupying" locations like Sinjajevina, Andrijevica, Berane and the border of Montenegro. The Municipality of Andrijevica has been deliberately chosen as a location for new NATO military barracks because of its proximity to the Albanian border. That is also the case with the old airport in Berane, which NATO plans to turn into its military base. The citizens of Montenegro need to resist this occupation and militarisation of the country.\n\n&nbsp;',
 'Ukraine participates in the investigation, despite the fact that Kiev violated international rules and did not close the airspace over the territory where the fighting took place.']

# for each of the data, send a query to gpt4o and get a response

from openai import OpenAI
client = OpenAI()

class QueryResponse(BaseModel):
    queries: list[str]

all_responses = []
for d in tqdm(data):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"""Here's a document, compose 20 queries partially related to the document: "{d}"."""
            }
        ],
        response_format=QueryResponse
    )

    structured_response = completion.choices[0].message.parsed
    # Write the structured response to a JSON file
    all_responses.append(structured_response.dict())

with open('queries.json', 'w') as f:
    json.dump(all_responses, f, indent=4)
