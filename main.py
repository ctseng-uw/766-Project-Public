from torchvision.utils import save_image
from tqdm import tqdm
import torch
import dataset
import config
import argparse


def test():
    channel = 1 if config.TO_GRAY else 3
    G_src2dst = config.MODEL.Generator(channel).to(config.DEVICE)
    G_dst2src = config.MODEL.Generator(channel).to(config.DEVICE)
    checkpoint = torch.load(config.CHECKPOINT_PATH)
    print(f"Loading checkpoint from {config.CHECKPOINT_PATH}")
    G_src2dst.load_state_dict(checkpoint["G_src2dst_state_dict"])
    G_dst2src.load_state_dict(checkpoint["G_dst2src_state_dict"])

    testset_loader = tqdm(
        dataset.get_dataloader(
            config.SRC_DATA,
            config.DST_DATA,
            config.BATCH_SIZE,
            shuffle=False,
            random_flip=False,
            to_gray=config.TO_GRAY,
        ),
        leave=True,
    )

    for idx, (src_img, dst_img, _, _) in enumerate(testset_loader):
        src_img = src_img.to(config.DEVICE)
        dst_img = dst_img.to(config.DEVICE)

        G_fake_dst_img = G_src2dst(src_img)
        G_fake_src_img = G_dst2src(dst_img)

        save_image(G_fake_dst_img, f"results/fakedst/{idx}.png")
        save_image(G_fake_src_img, f"results/fakesrc/{idx}.png")


def train():
    channel = 1 if config.TO_GRAY else 3
    G_src2dst = config.MODEL.Generator(channel).to(config.DEVICE)
    G_dst2src = config.MODEL.Generator(channel).to(config.DEVICE)
    D_src = config.MODEL.Discriminator(channel).to(config.DEVICE)
    D_dst = config.MODEL.Discriminator(channel).to(config.DEVICE)

    mse = torch.nn.MSELoss()

    dis_optimizer = torch.optim.Adam(
        list(D_src.parameters()) + list(D_dst.parameters()),
        lr=config.LR,
        betas=(0.5, 0.99),
    )

    gen_optimizer = torch.optim.Adam(
        list(G_dst2src.parameters()) + list(G_src2dst.parameters()),
        lr=config.LR,
        betas=(0.5, 0.99),
    )

    if config.LOAD_CHECKPOINT and config.CHECKPOINT_PATH:
        print(f"Loading checkpoint from {config.CHECKPOINT_PATH}")
        checkpoint = torch.load(config.CHECKPOINT_PATH)
        G_src2dst.load_state_dict(checkpoint["G_src2dst_state_dict"])
        G_dst2src.load_state_dict(checkpoint["G_dst2src_state_dict"])
        epoch_st = checkpoint["epoch"]
        D_src.load_state_dict(checkpoint["D_src_state_dict"])
        D_dst.load_state_dict(checkpoint["D_dst_state_dict"])
        dis_optimizer.load_state_dict(checkpoint["dis_optimizer"])
        gen_optimizer.load_state_dict(checkpoint["gen_optimizer"])
        print(f"Starting from epoch {epoch_st}")
    else:
        epoch_st = -1

    gen_scaler = torch.cuda.amp.GradScaler()
    dis_scaler = torch.cuda.amp.GradScaler()
    for epoch_idx in range(epoch_st + 1, config.EPOCH):
        trainset_loader = tqdm(
            dataset.get_dataloader(
                config.SRC_DATA,
                config.DST_DATA,
                config.BATCH_SIZE,
                shuffle=True,
                random_flip=config.RANDOM_FLIP,
                to_gray=config.TO_GRAY,
            ),
            leave=True,
        )

        # D_Loss_sum = L_sum = 0
        for idx, (src_img, dst_img, src_img2, dst_img2) in enumerate(trainset_loader):
            src_img = src_img.to(config.DEVICE)
            dst_img = dst_img.to(config.DEVICE)

            G_fake_dst_img = G_src2dst(src_img)
            G_fake_src_img = G_dst2src(dst_img)
            D_real_dst_img = D_dst(dst_img)
            D_real_src_img = D_src(src_img)
            D_fake_dst_img = D_dst(G_fake_dst_img.detach())
            D_fake_src_img = D_src(G_fake_src_img.detach())

            # Train Discriminators
            # For discriminators, L = (1 - D(real_img)) ^ 2 + D(fake_img)^2
            L_D_src = mse(D_real_src_img, torch.ones_like(D_real_src_img)) + mse(
                D_fake_src_img, torch.zeros_like(D_fake_src_img)
            )
            L_D_dst = mse(D_real_dst_img, torch.ones_like(D_real_dst_img)) + mse(
                D_fake_dst_img, torch.zeros_like(D_fake_dst_img)
            )
            D_Loss = (L_D_src + L_D_dst) / 2

            dis_optimizer.zero_grad()
            dis_scaler.scale(D_Loss).backward()
            dis_scaler.step(dis_optimizer)
            dis_scaler.update()

            # ==Train Generators==#
            # For generators, the adverserial loss L = (1 - D(fake_img)) ^ 2
            D_fake_src_img = D_src(G_fake_src_img)
            D_fake_dst_img = D_dst(G_fake_dst_img)
            L_G_src2dst = mse(D_fake_dst_img, torch.ones_like(D_fake_dst_img))
            L_G_dst2src = mse(D_fake_src_img, torch.ones_like(D_fake_src_img))
            L_adversial_loss = L_G_src2dst + L_G_dst2src

            # Cycle loss
            # Cycle src: src -> dst -> src
            if config.LAMBDA_CYCLE != 0:
                cycle_src = G_dst2src(G_fake_dst_img)
                L_cycle_src = mse(cycle_src, src_img)
                cycle_dst = G_src2dst(G_fake_src_img)
                L_cycle_dst = mse(cycle_dst, dst_img)
                L_cycle_loss = L_cycle_src + L_cycle_dst
            else:
                L_cycle_loss = 0

            # Identity Loss
            # Real image to its generator will generate an output like its input
            if config.LAMBDA_IDENTITY != 0:
                identity_src = G_dst2src(src_img)
                identity_dst = G_src2dst(dst_img)
                L_identity_src = torch.nn.L1Loss()(identity_src, src_img)
                L_identity_dst = torch.nn.L1Loss()(identity_dst, dst_img)
                L_identity_loss = L_identity_src + L_identity_dst
            else:
                L_identity_loss = 0

            # True Loss
            if config.LAMBDA_TRUE != 0:
                dst_img2 = dst_img2.to(config.DEVICE)
                src_img2 = src_img2.to(config.DEVICE)
                L_true_loss = mse(G_fake_src_img, src_img2) + mse(
                    G_fake_dst_img, dst_img2
                )
            else:
                L_true_loss = 0

            # Loss of generators: adverserial loss + cycle loss + identity loss
            L = (
                L_adversial_loss
                + L_cycle_loss * config.LAMBDA_CYCLE
                + L_identity_loss * config.LAMBDA_IDENTITY
                + L_true_loss * config.LAMBDA_TRUE
            )
            gen_optimizer.zero_grad()
            gen_scaler.scale(L).backward()
            gen_scaler.step(gen_optimizer)
            gen_scaler.update()

            if idx % 100 == 0:
                save_image(G_fake_dst_img, f"tmp_imgs/curret_dst_font_{idx}.png")
                save_image(G_fake_src_img, f"tmp_imgs/curret_src_font_{idx}.png")

        saved_dst_img = torch.cat((src_img, G_fake_dst_img, dst_img2), 3)
        saved_src_img = torch.cat((dst_img, G_fake_src_img, src_img2), 3)

        save_image(saved_dst_img, f"tmp_imgs/dst_font_{epoch_idx}.png")
        save_image(saved_src_img, f"tmp_imgs/src_font_{epoch_idx}.png")

        print(f"epoch_idx:{epoch_idx} - Loss of discriminator:{D_Loss}")
        print(f"epoch_idx:{epoch_idx} - Loss of generators:{L}")

        if epoch_idx % 5 == 0 or epoch_idx == config.EPOCH - 1:
            print(f"save torch:{epoch_idx}")
            torch.save(
                {
                    "epoch": epoch_idx,
                    "G_src2dst_state_dict": G_src2dst.state_dict(),
                    "G_dst2src_state_dict": G_dst2src.state_dict(),
                    "D_dst_state_dict": D_dst.state_dict(),
                    "D_src_state_dict": D_src.state_dict(),
                    "dis_optimizer": dis_optimizer.state_dict(),
                    "gen_optimizer": gen_optimizer.state_dict(),
                },
                f"checkpoints/checkpoints_{epoch_idx}",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        test()
    else:
        train()
