system_prompt="""
You are an expert in NMR spectroscopy analysis and organic chemistry.
"""

def user_prompt(NMR_data):
    return f"""
            Here is HNMR spectrum data:
            {NMR_data}
            What's the molecule name? 
            Think step-by-step, making extensive use of a scratchpad to record your thoughts. Consider finding ways to group related peaks 
            together, and keep track of the stoichiometry and the amount of unassigned H atoms as you make provisional assignments.
            Please solely look at the HNMR to determine molecule.
            Format the final answer like this - 
            ### Scratchpad ### <scratchpad> ### Scratchpad ###
            ### Start answer ### <prediction> ### End answer ###
            The prediction should only contain the name of the molecule and no other text
        """