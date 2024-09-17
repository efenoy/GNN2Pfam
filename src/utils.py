
def train_one_epoch(epoch_index, tb_writer, model, train_loader, optimizer, loss_fn, device):
    # running_loss = 0.
    last_loss = 0.
    model.train()
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(train_loader):

        inputs, edge_index, batch_idx = data.x.to(device), data.edge_index.to(device), data.batch.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs, edge_index, batch_idx)
        pred = outputs.squeeze(dim=-1)
        pred = (pred > 0).float()

        labels = data.y.float().to(device)
        # Compute the loss and its gradients
        loss = loss_fn(pred, labels)
        loss.requires_grad = True
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        last_loss=loss.item()
        tb_writer.add_scalar('Loss/train', last_loss, epoch_index)
        # Gather data and report
        # running_loss += loss.item()
        # if i % 100 == 99:
        #     last_loss = running_loss / 100 # loss per batch
        #     print('  batch {} loss: {}'.format(i + 1, last_loss))
        #     tb_x = epoch_index * len(train_loader) + i + 1
        #     tb_writer.add_scalar('Loss/train', last_loss, tb_x)
        #     running_loss = 0.

    return last_loss
