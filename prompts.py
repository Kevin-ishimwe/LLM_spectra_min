system_prompt="""You are an expert in NMR spectroscopy analysis and organic chemistry."""

# 1.zero-shot prompt
def base_prompt(NMR_data,formula=None):
    return f"""
            Here is 1H NMR spectrum data for a certain molecule {":" if formula is None else "with chemical formula {formula}:"}
            {NMR_data}
            What's the molecule's name?
            {"" if formula is None else "Be mindful of stoichiometry and ensure consistency with the given formula." }
            Format the final answer like this - 
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
        """

# 2.chain of thought prompt 
def base_COT(NMR_data,formula=None):
    return f"""
            Here is 1H NMR spectrum data for a certain molecule {":" if formula is None else "with chemical formula {formula}:"}
            {NMR_data}
            What's the molecule's name?
            Think step-by-step, making extensive use of a scratchpad to record your thoughts.
            Consider finding ways to group related peaks 
            together, and keep track of the stoichiometry and the amount of unassigned H atoms as you make provisional assignments.
            {"" if formula is None else "Be mindful of stoichiometry and ensure consistency with the given formula." }
            Format the final answer like this - 
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
        """

# 3.chain of thought prompt elaborate (logic)
def logic_tips_COT(NMR_data,formula=None):
    return f"""
           Here is 1H NMR spectrum data for a certain molecule {":" if formula is None else "with chemical formula {formula}:"}
            {NMR_data}
            What's the molecule's name?
            Let's think step-by-step. 
            Use a scratchpad to record your thoughts. First, identify all groups of peaks and the corresponding number of hydrogens, and ensure the stoichiometry of the formula matches that in the spectrum Then, for each group of peaks, hypothesize one or more molecule fragments that could explain the signals in that peak.
            Rank the hypotheses according to which is most plausible
            Occasionally synthesize a candidate molecule from the provisionally-assigned fragments
            Be mindful of connectivity and ensure consistency with all hypothesized fragments
            {"" if formula is None else "Be mindful of stoichiometry and ensure consistency with the given formula." }
            Continue using the scratchpad until you're confident about the answer.
            Format the final answer like this - 
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
            
        """

#  4.chain of thought prompt elaborate( expert tips )  
def expert_tips_COT(NMR_data,formula=None):
    return f"""
            Here is 1H NMR spectrum data for a certain molecule {":" if formula is None else "with chemical formula {formula}:"}
            {NMR_data} 
            What's the molecule's name? Let's think step-by-step. 
            Use a scratchpad to record your thoughts, and continuing using it until your confident about your answer. 
            
            Before the analysis of NMR spectra, think about the molecular formula. Is the molecule fully saturated, or are there some multiple bonds or rings?
            Consider the solvent used. Do you expect the appearance of signals from exchangeable hydrogens (e.g., in CDCl3or DMSO), or are they exchanged to deuterium (e.g. in D2O or methanol-d4)? 
            Identify solvent signals and ignore them in the structural analysis.
            Consider the number of signals, their positions (chemical shifts), intensity, shape and splitting.
            The number of signals corresponds to the number of non-equivalent atoms in the molecule. For example, benzene will have only one carbon signal and one hydrogen signal because all carbon atoms in its structure are equivalent and all hydrogen atoms are equivalent.
            Typical regions in proton NMR spectra are: carboxylic acids: 10–13 ppm, aldehydes: 9–11 ppm, aromatic hydrogens: 6.5–8 ppm, hydrogens attached to a double-bond carbon: 4–7.5 ppm, O–CH fragments (alcohols, ethers): 3–5 ppm, hydrogens attached to a triple-bond carbon: 2–3.5 ppm, hydrogens attached to a single-bond carbon without an electronegative substituent: 0.5–2.5 ppm.
            The intensity of signals reflects the number of equivalent atoms contributing to the signal. Hydrogen spectra can usually be quantified by integration. The integral intensities are shown below the signals. Note, however, that slight deviations of the integral intensities from ideal values can be expected. Unfortunately, signal intensities in commonly measured carbon spectra can be significantly affected by other effects than the number of equivalent atoms and cannot be quantified.
            The shape of signals may reflect dynamic processes, such as conformational equilibria. Broad signals are often observed for systems with slow dynamics.
            Exchangeable hydrogen atoms are those attached to an oxygen, nitrogen or sulfur atom. Signals of exchangeable protons are often broader than other signals.
            The splitting of signals caused by J-coupling reveals the number of (mostly) hydrogen atoms in neighboring positions. One neighboring hydrogen splits the signal into two lines (doublet) with 1:1 ratio of signal intensities, two neighboring hydrogens split the signal into three lines (triplet) with 1:2:1 intensity ratio, three neighboring hydrogens split the signal into four lines (quartet) with 1:3:3:1 intensity ratio. For example, the signal of the methyl group in an ethoxy fragment (CH3–CH2–O) is split into a triplet by the neighboring two hydrogen atoms in the CH2 group. Similarly, the signal of the CH2 group is split into a quartet by the neighboring three hydrogen atoms in the methyl group.
            
            {"" if formula is None else "Be mindful of stoichiometry and ensure consistency with the given formula." }  
            Format the final answer like this - 
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text

    """
# 5.chain of thought prompt elaborate (logic+expert tips)
def expert_logics_tips_COT(NMR_data,formula=None):
    return f"""
            Here is 1H NMR spectrum data for a certain molecule {":" if formula is None else "with chemical formula {formula}:"}
            {NMR_data} 
            Let's think step-by-step. 
            Use a scratchpad to record your thoughts, and continuing using it until your confident about your answer. 
            First, identify all groups of peaks and the corresponding number of hydrogens, and ensure the stoichiometry of the formula matches that in the spectrum. Then, for each group of peaks, hypothesize one or more molecule fragments that could explain the signals in that peak.
            Rank the hypotheses according to which is most plausible.
            Occasionally synthesize a candidate molecule from the provisionally-assigned fragments.
             
            Before the analysis of NMR spectra, think about the molecular formula. 
            Is the molecule fully saturated, or are there some multiple bonds or rings?
            Consider the solvent used. Do you expect the appearance of signals from exchangeable hydrogens (e.g., in CDCl3or DMSO), or are they exchanged to deuterium (e.g. in D2O or methanol-d4)? 
            Identify solvent signals and ignore them in the structural analysis.
            Consider the number of signals, their positions (chemical shifts), intensity, shape and splitting.
            The number of signals corresponds to the number of non-equivalent atoms in the molecule. For example, benzene will have only one carbon signal and one hydrogen signal because all carbon atoms in its structure are equivalent and all hydrogen atoms are equivalent.
            Typical regions in proton NMR spectra are: carboxylic acids: 10–13 ppm, aldehydes: 9–11 ppm, aromatic hydrogens: 6.5–8 ppm, hydrogens attached to a double-bond carbon: 4–7.5 ppm, O–CH fragments (alcohols, ethers): 3–5 ppm, hydrogens attached to a triple-bond carbon: 2–3.5 ppm, hydrogens attached to a single-bond carbon without an electronegative substituent: 0.5–2.5 ppm.
            The intensity of signals reflects the number of equivalent atoms contributing to the signal. Hydrogen spectra can usually be quantified by integration. The integral intensities are shown below the signals. Note, however, that slight deviations of the integral intensities from ideal values can be expected. 
            The shape of signals may reflect dynamic processes, such as conformational equilibria. Broad signals are often observed for systems with slow dynamics.
            Exchangeable hydrogen atoms are those attached to an oxygen, nitrogen or sulfur atom. Signals of exchangeable protons are often broader than other signals.
            The splitting of signals caused by J-coupling reveals the number of (mostly) hydrogen atoms in neighboring positions. One neighboring hydrogen splits the signal into two lines (doublet) with 1:1 ratio of signal intensities, two neighboring hydrogens split the signal into three lines (triplet) with 1:2:1 intensity ratio, three neighboring hydrogens split the signal into four lines (quartet) with 1:3:3:1 intensity ratio. For example, the signal of the methyl group in an ethoxy fragment (CH3–CH2–O) is split into a triplet by the neighboring two hydrogen atoms in the CH2 group. Similarly, the signal of the CH2 group is split into a quartet by the neighboring three hydrogen atoms in the methyl group. 
            
            Be mindful of connectivity and ensure consistency with all hypothesized fragments. 
            {"" if formula is None else "Be mindful of stoichiometry and ensure consistency with the given formula." }

            Format the final answer like this - 
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
    """


# self augmentation prompt
def consistency_reprompt():
    return """
            is the predicted molecule name consistent with the formula, answer with 
            ### Scratchpad ###
            ### Start answer ###<yes or no>### End answer ###
            """


def regeneration_prompt():
    return f"""
            re-analyse the spectra data
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
            """
