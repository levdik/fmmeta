import os

if __name__ == '__main__':
    for wavelength in [650, 550, 450]:
        for period in [400, 300]:
            os.system(f'start cmd /k "venv\\Scripts\\activate && python design_monochromatic.py {wavelength} {period}"')
