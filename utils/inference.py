import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Maximum_likelihood_estimate(input, model, tokenizer, num_beams=1000, device=device):
    """
    Return model MLE by performing beamsearch over output tokens.

    Parameters
    ----------
    input : str
        Formatted input, (abstract + categorie), for generating titles
    model : Pytorch language model
        The conditional language model used to generate output
    tokenizer : Pytorch Transformer Tokenizer
        Tokenizer used to transform and inverse-transform tokens to text 
    num_beams : int, optional
        How many beams to track using beamsearch, by default 1000
    device : str
        Flag to run inference on cuda or cpu

    returns
    -------
    MLE_output : list-tokens
        tokenized output prediction
    """
    input = tokenizer(f"{input}", return_tensors="pt").input_ids.to(device)

    MLE_output = model.generate(
        input,
        max_length=1024,
        num_return_sequences=1,
        early_stopping=True,
        num_beams=num_beams,
    )
    return [MLE_output]


def random_sampling(
    input, model, tokenizer, N=10, top_p=0.9, temperature=1.5, device=device
):
    """
    Generate outputs via random sampling, using the nucleus method.
    top_p selects the smallest number of outputs tokens that together have probability p, and uses this set to sample.
    Temperature is analogues to temperature in statistical mechanics, and reweights the distribution in such a manner that
    less likely events become more likely.

    Parameters
    ----------
    input : str
        Formatted input, (abstract + categorie), for generating titles
    model : Pytorch language model
        The conditional language model used to generate output
    tokenizer : Pytorch Transformer Tokenizer
        Tokenizer used to transform and inverse-transform tokens to text
    N : int, optional
        How many outputs to generate, by default 10
    top_p : float, optional
        Nucleus sampling value, by default 0.9
    temperature : float, optional
        how to reweight the token probabilities, 1 means nothing happens, by default 1.5
    device : str
        Flag to run inference on cuda or cpu

    Returns
    -------
    random_outputs : list
        list with N output predictions 
    """
    input = tokenizer(f"{input}", return_tensors="pt").input_ids.to(device)

    random_outputs = model.generate(
        input,
        max_length=1024,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=N,
        early_stopping=True,
    )

    return random_outputs
