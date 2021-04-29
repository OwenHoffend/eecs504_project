import matplotlib.pyplot as plt

def pl(fn, c):
    with open(fn, 'r', encoding="utf-8") as f:
        epoch_cnt = 0
        tr_losses = []
        val_losses = []
        tr_ious = []
        r_avg_len = 10
        r_avgs = [0 for _ in range(r_avg_len)]
        for line in f:
            if 'Epoch' in line:
                epoch_cnt += 1
            if 'Avg Overall Loss' in line:
                loss = float(line.split(': ')[2])
                if 'training' in line:
                    tr_losses.append(loss)
                else:
                    val_losses.append(loss)
            if 'IOU' in line and not ('nan' in line):
                iou = float(line.split(' ')[10])
                r_avgs[epoch_cnt % r_avg_len] = iou
                if epoch_cnt > r_avg_len:
                    tr_ious.append(sum(r_avgs) / r_avg_len)
                else:
                    tr_ious.append(iou)

        xvals = list(range(len(tr_losses)))
        plt.plot(xvals, tr_losses, color=c, label='{}: Train loss'.format(fn))
        #plt.plot(xvals, val_losses, color=c, label='{}: Val loss'.format(fn))

        #plt.plot(list(range(len(tr_ious))), tr_ious, label='{}: Train IoUs'.format(fn))
        #plt.plot(xvals, val_ious, label='{}: Val IoUs'.format(fn))

if __name__ == '__main__':
    pl('output.txt', 'tab:blue')
    pl('output2.txt', 'tab:green')
    plt.xlabel("Epoch Number")
    plt.ylabel("Avg IoU")
    plt.title("IoU (10-sample running avg.) vs. Epoch Number")
    plt.legend()
    plt.show()