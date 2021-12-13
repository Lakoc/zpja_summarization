from transformers import pipeline


class T5Summary:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

    def summarize(self, document, min_len, max_len):
        summary_text = self.summarizer(document, max_length=max_len, min_length=min_len, do_sample=False)[0][
            'summary_text']
        return summary_text


if __name__ == '__main__':
    t5 = T5Summary()
    text= """
    Coronaviruses are a group of related RNA viruses that cause diseases in mammals and birds. In humans and birds, they cause respiratory tract infections that can range from mild to lethal. Mild illnesses in humans include some cases of the common cold (which is also caused by other viruses, predominantly rhinoviruses), while more lethal varieties can cause SARS, MERS and COVID-19. In cows and pigs they cause diarrhea, while in mice they cause hepatitis and encephalomyelitis.

Coronaviruses constitute the subfamily Orthocoronavirinae, in the family Coronaviridae, order Nidovirales and realm Riboviria.[3][4] They are enveloped viruses with a positive-sense single-stranded RNA genome and a nucleocapsid of helical symmetry.[5] The genome size of coronaviruses ranges from approximately 26 to 32 kilobases, one of the largest among RNA viruses.[6] They have characteristic club-shaped spikes that project from their surface, which in electron micrographs create an image reminiscent of the solar corona, from which their name derives.[7]

The name "coronavirus" is derived from Latin corona, meaning "crown" or "wreath", itself a borrowing from Greek κορώνη korṓnē, "garland, wreath".[8][9] The name was coined by June Almeida and David Tyrrell who first observed and studied human coronaviruses.[10] The word was first used in print in 1968 by an informal group of virologists in the journal Nature to designate the new family of viruses.[7] The name refers to the characteristic appearance of virions (the infective form of the virus) by electron microscopy, which have a fringe of large, bulbous surface projections creating an image reminiscent of the solar corona or halo.[7][10] This morphology is created by the viral spike peplomers, which are proteins on the surface of the virus.[11]

The scientific name Coronavirus was accepted as a genus name by the International Committee for the Nomenclature of Viruses (later renamed International Committee on Taxonomy of Viruses) in 1971.[12] As the number of new species increased, the genus was split into four genera, namely Alphacoronavirus, Betacoronavirus, Deltacoronavirus, and Gammacoronavirus in 2009.[13] The common name coronavirus is used to refer to any member of the subfamily Orthocoronavirinae.[4] As of 2020, 45 species are officially recognised.[14]
    """

    # text = """One month after the United States began what has become a troubled rollout of a national COVID vaccination campaign, the effort is finally gathering real steam.
    # Close to a million doses -- over 951,000, to be more exact -- made their way into the arms of Americans in the past 24 hours, the U.S. Centers for Disease Control and Prevention reported Wednesday. That's the largest number of shots given in one day since the rollout began and a big jump from the previous day, when just under 340,000 doses were given, CBS News reported.
    # That number is likely to jump quickly after the federal government on Tuesday gave states the OK to vaccinate anyone over 65 and said it would release all the doses of vaccine it has available for distribution. Meanwhile, a number of states have now opened mass vaccination sites in an effort to get larger numbers of people inoculated, CBS News reported."""
    summary = t5.summarize(text, min_len=5, max_len=100)

    print(summary)
