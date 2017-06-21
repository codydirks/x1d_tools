

class SpectralLine(object):
    """
    Parses and stores individual atomic line info for entries in FITS6P's atomic.dat file.
    """
    def __init__(self, atomic_dat_line):
		line=atomic_dat_line.strip('\n')
		self.lam=float(line[0:10])
		self.ion=line[10:19].strip()
		self.a=int(line[19:21].strip())
		self.w=int(line[21:24].strip())
		self.i=int(line[24:26].strip())
		self.m=int(line[26:28].strip())
		self.f=line[28:38].strip()
		if self.f != '':
			self.f=float(self.f)
		else:
			self.f=0.0

		self.gamma=line[38:48].strip()
		if self.gamma != '':
			self.gamma=float(self.gamma)
		else:
			self.gamma=0.0

		self.notes=line[48:].strip()

    def __repr__(self):
		return '{:<10}'.format(self.lam) + str(self.ion)

    def __str__(self):
		return self.__repr__()

    def __eq__(self,other):
        """
        Overloaded boolean equality operator. Checks that both the ion name and wavelength are identical.
        """
        if (self.lam == other.lam) and (self.ion == other.ion):
        	return True
        else:
        	return False
