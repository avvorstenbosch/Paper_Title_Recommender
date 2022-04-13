def tokens_to_text(results_tokenized, tokenizer):
    """
    Transform embedding tokens back to regular text

    Parameters
    ----------
    results_tokenized : list - Tokenized output
        Raw output prediction from model
    tokenizer : Pytorch Transformer Tokenizer
        Tokenizer used to transform and inverse-transform tokens to text

    Returns
    -------
    results : list
        list containing output in string format
    """
    results = [
        tokenizer.decode(result_tokenized, skip_special_tokens=True)
        for result_tokenized in results_tokenized
    ]
    return results


def pretty_print():
    """
    Generate nicely formated result for printing and saving
    """
    pass
