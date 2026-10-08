# TransitBox

Website: [https://ccoopq.github.io/TransitBox/](https://ccoopq.github.io/TransitBox/)

TransitBox is a web dashboard for bus passenger tracking, boarding/alighting matching, and payment visualization, built on GHR-VLM and TransReID.

## Deployment

Requires Python 3.

```bash
git clone https://github.com/ccoopq/TransitBox.git
cd TransitBox
python3 serve.py --directory . --host 0.0.0.0 --port 8787
```

Open [http://localhost:8787](http://localhost:8787).

Videos and passenger data are available on the website, with faces blurred, and are not included in this repository.
