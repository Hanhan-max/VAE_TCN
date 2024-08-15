import numpy as np
import torch
from tqdm import tqdm


def trainer(num_epochs, train_dataloader, eval_dataloader,model,criterion,optimizer,input_len,device,scheduler):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        losses_train = []
        for i, item in enumerate(train_dataloader):
            input = item[0][:, :input_len].to(device)
            traget = item[0][:, input_len:].to(device)
            out = model(input)
            loss = criterion(out, traget)
            losses_train.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        tqdm.write(f"\t Epoch {epoch + 1} / {num_epochs}, Loss: {(sum(losses_train) / len(losses_train)):.6f}")
        tqdm.write(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for i, item in enumerate(eval_dataloader):
                    input = item[0][:, :input_len].to(device)
                    traget = item[0][:, input_len:].to(device)
                    out = model(input)
                    loss = criterion(out, traget)
                    losses.append(loss)
                    eval_loss = sum(losses) / len(losses)
                print(f'\nEpoch [{epoch + 1}/{num_epochs}], Eval_Loss: {eval_loss:.4f}')
        scheduler.step()
def decoder_trainer(num_epochs, train_dataloader, eval_dataloader,model,criterion,optimizer,input_len,decode_len,device):
    for epoch in tqdm(range(num_epochs)):
        model.train()
        losses_train = []
        for i, item in enumerate(train_dataloader):
            input = item[0][:, :input_len].to(device)
            decoder_input = item[0][:, decode_len:input_len].to(device)
            traget = item[0][:, input_len:].to(device)

            decoder_input_traget = torch.zeros_like(traget)
            decoder_input = torch.cat((decoder_input, decoder_input_traget), dim=1).to(device)
            out = model(input,decoder_input)
            out = out[:,(input_len-decode_len):,:]
            loss = criterion(out, traget)
            losses_train.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        tqdm.write(f"\t Epoch {epoch + 1} / {num_epochs}, Loss: {(sum(losses_train) / len(losses_train)):.6f}")
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                losses = []
                for i, item in enumerate(eval_dataloader):
                    input = item[0][:, :input_len].to(device)
                    decoder_input = item[0][:, decode_len:input_len].to(device)
                    traget = item[0][:, input_len:].to(device)

                    decoder_input_traget = torch.zeros_like(traget)
                    decoder_input = torch.cat((decoder_input, decoder_input_traget), dim=1).to(device)

                    out = model(input, decoder_input)
                    out = out[:, (input_len - decode_len):, :]
                    loss = criterion(out, traget)
                    losses.append(loss)
                print(f'\nEpoch [{epoch + 1}/{num_epochs}], Eval_Loss: {sum(losses) / len(losses) :.4f}')



def test(model,test_dataloader,input_len,device):
    model.eval()
    result = []
    true = []
    with torch.no_grad():
        for i, item in enumerate(test_dataloader):
            input = item[0][:, :input_len].to(device)
            traget = item[0][:, input_len:]
            out = model(input)
            result.append(out.detach().cpu().numpy())
            true.append(traget)
    result_np = np.concatenate(result, axis=0)
    true_np = np.concatenate(true, axis=0)
    return result_np,true_np


def decoder_test(model,test_dataloader,input_len,decode_len,device):
    model.eval()
    result = []
    true = []
    with torch.no_grad():
        for i, item in enumerate(test_dataloader):
            input = item[0][:, :input_len].to(device)
            decoder_input = item[0][:, decode_len:input_len].to(device)
            traget = item[0][:, input_len:].to(device)

            decoder_input_traget = torch.zeros_like(traget)
            decoder_input = torch.cat((decoder_input, decoder_input_traget), dim=1).to(device)

            out = model(input, decoder_input)
            out = out[:, (input_len - decode_len):, :]
            result.append(out.detach().cpu().numpy())
            true.append(traget.detach().cpu().numpy())
    result_np = np.concatenate(result, axis=0)
    true_np = np.concatenate(true, axis=0)
    return result_np,true_np