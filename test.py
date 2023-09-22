import argparse
import torch
from models import Generator
# from utils import save_images


def predict(args):
    generator = Generator().to(args.device)
    generator.load_state_dict(torch.load("./weights/generator_stage4_epoch950.pth"),strict=False)
    generator.eval()

    with torch.no_grad():
        fix_latent_dim = min(args.base_channels * 2 ** args.num_stages-128, 512)
        fix_z = torch.randn((args.batch_size, fix_latent_dim, 1, 1, 1),device=args.device) # 固定噪声维度扩展并赋值为高斯分布
        # z = torch.randn(args.batch_size, latent_dim, 1, 1, device=args.device)
        porosity_label = 0.2
        
        porosity_label = porosity_label.view(args.batch_size, 1, 1, 1, 1) # 标签扩展为5D
        dporosity_label = porosity_label.expand(args.batch_size, 128, 1, 1, 1) # 标签在通道维度上复制128次
        dporosity_label = dporosity_label.type(torch.cuda.FloatTensor) # 将标签转换为FloatTensor类型
        
        # 生成假图像
        with torch.no_grad():
            fake_images = Generator(dporosity_label, fix_z, 1, args.num_stages)
        
        # output = generator(z, 1, args.num_stages)
        # output = output.permute(0, 2, 3, 1).cpu().numpy()
        # save_images("./output/g_0.15", fake_images)


def main():
    parser = argparse.ArgumentParser(description="PGGAN")
    parser.add_argument("--num_stages", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default='./output/g_0.15')
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    predict(args)

if __name__ == '__main__':
    main()
