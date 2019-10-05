from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

model_file_url = 'https://drive.google.com/open?id=1-4Byho2Gs32FFBiI6sbV_kKjImXoMsYt'
model_file_name = 'flowers-stage-2'
classes = ['pink primrose', 
'hard-leaved pocket orchid', 
'canterbury bells', 
'sweet pea', 
'english marigold', 
'tiger lily', 
'moon orchid', 
'bird of paradise', 
'monkshood', 
'globe thistle', 
'snapdragon', 
'colts foot', 
'king protea', 
'spear thistle', 
'yellow iris', 
'globe-flower', 
'purple coneflower', 
'peruvian lily', 
'balloon flower', 
'giant white arum lily', 
'fire lily', 
'pincushion flower', 
'fritillary', 
'red ginger', 
'grape hyacinth', 
'corn poppy', 
'prince of wales feathers', 
'stemless gentian', 
'artichoke', 
'sweet william', 
'carnation', 
'garden phlox', 
'love in the mist', 
'mexican aster', 
'alpine sea holly', 
'ruby-lipped cattleya', 
'cape flower', 
'great masterwort', 
'siam tulip', 
'lenten rose', 
'barbeton daisy', 
'daffodil', 
'sword lily', 
'poinsettia', 
'bolero deep blue', 
'wallflower', 
'marigold', 
'buttercup', 
'oxeye daisy', 
'common dandelion', 
'petunia', 
'wild pansy', 
'primula', 
'sunflower', 
'pelargonium', 
'bishop of llandaff', 
'gaura', 
'geranium', 
'orange dahlia', 
'pink-yellow dahlia', 
'cautleya spicata', 
'japanese anemone', 
'black-eyed susan', 
'silverbush', 
'californian poppy', 
'osteospermum', 
'spring crocus', 
'bearded iris', 
'windflower', 
'tree poppy', 
'gazania', 
'azalea', 
'water lily', 
'rose', 
'thorn apple', 
'morning glory', 
'passion flower', 
'lotus', 
'toad lily', 
'anthurium', 
'frangipani', 
'clematis', 
'hibiscus', 
'columbine', 
'desert-rose', 
'tree mallow', 
'magnolia', 
'cyclamen', 
'watercress', 
'canna lily', 
'hippeastrum', 
'bee balm', 
'ball moss', 
'foxglove', 
'bougainvillea', 
'camellia', 
'mallow', 
'mexican petunia', 
'bromelia', 
'blanket flower', 
'trumpet creeper', 
'blackberry lily']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(model_file_url, path/'models'/f'{model_file_name}.pth')
    data_bunch = ImageDataBunch.single_from_classes(path, classes,
        tfms=get_transforms(), size=224).normalize(imagenet_stats)
    learn = cnn_learner(data_bunch, models.resnet34, pretrained=False)
    learn.load(model_file_name)
    return learn

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    return JSONResponse({'result': learn.predict(img)[0]})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app, host='0.0.0.0', port=8080)

