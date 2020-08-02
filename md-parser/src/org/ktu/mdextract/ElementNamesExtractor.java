package org.ktu.mdextract;

import com.nomagic.magicdraw.core.Application;
import com.nomagic.magicdraw.core.Project;
import com.nomagic.magicdraw.uml.symbols.DiagramPresentationElement;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Association;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Diagram;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Element;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.NamedElement;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Package;
import com.nomagic.uml2.ext.magicdraw.classes.mdkernel.Type;
import com.nomagic.uml2.ext.magicdraw.mdusecases.Actor;
import com.nomagic.uml2.ext.magicdraw.mdusecases.UseCase;
import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;


public class ElementNamesExtractor {

    protected Collection<DiagramPresentationElement> diagrams;
    protected Collection<Element> candidateElements = new HashSet<>();
    protected Map<DiagramPresentationElement, Collection<Element>> elementsByDiagram = new HashMap<>();

    public ElementNamesExtractor(Package root) {
        this.diagrams = getDiagrams(root);
        for (DiagramPresentationElement diagram: diagrams)
            readElements(diagram);
    }

    public ElementNamesExtractor(Collection<DiagramPresentationElement> diagrams) {
        this.diagrams = diagrams;
        for (DiagramPresentationElement diagram: diagrams)
            readElements(diagram);
    }

    protected void readElements(DiagramPresentationElement diagram) {
        Collection<Element> diagramElements = diagram.getUsedModelElements();
        candidateElements.addAll(diagramElements);
        elementsByDiagram.put(diagram, diagramElements);
    }

    protected Collection<Element> getDiagramElements(Collection<Element> colelem, DiagramPresentationElement diagram) {
        Collection<Element> newelem = new HashSet<>();
        for (Element element : colelem)
            if (element instanceof Package && diagram.findPresentationElement(element, null) != null)
                addPackageElements(diagram, (Package) element, newelem);
        return newelem;
    }

    protected void addPackageElements(DiagramPresentationElement diagram, Package pack, Collection<Element> elements) {
        for (Element el : pack.getOwnedElement())
            if (diagram.findPresentationElement(el, null) != null)
                elements.add(el);
        for (Package innerPack : pack.getNestedPackage())
            if (diagram.findPresentationElement(innerPack, null) != null)
                addPackageElements(diagram, innerPack, elements);
    }

    protected Collection<DiagramPresentationElement> getDiagrams(Package model, Collection<DiagramPresentationElement> diagrams) {
        Project project = Application.getInstance().getProject();
        if (project == null || model == null)
            return diagrams;
        for (Diagram diag : model.getOwnedDiagram()) {
            DiagramPresentationElement pres = project.getDiagram(diag);
            if (pres != null)
                diagrams.add(pres);
        }
        for (Package pkg : model.getNestedPackage())
            diagrams = getDiagrams(pkg, diagrams);
        return diagrams;
    }

    public Collection<DiagramPresentationElement> getDiagrams(Package root) {
        Collection<DiagramPresentationElement> diagrams = new HashSet<>();
        return getDiagrams(root, diagrams);
    }

    private Set<String[]> extractAssociations(Collection<Element> candidateElements) {
        Set<Association> associations = candidateElements.stream()
                .filter(el -> el.getClassType().equals(Association.class))
                .filter(el -> {
                    Collection<Type> endtypes = ((Association) el).getEndType();
                    boolean actor_found = false, uc_found = false;
                    for (Type elem : endtypes)
                        if (elem.getClassType().equals(Actor.class))
                            actor_found = true;
                        else if (elem.getClassType().equals(UseCase.class))
                            uc_found = true;
                    return actor_found && uc_found;
                }).map(e -> (Association)e)
                .collect(Collectors.toSet());
        return associations.stream()
                .map(el -> {
                    Collection<Type> endtypes = el.getEndType();
                    String actorName = null, ucName = null;
                    for (Type elem : endtypes)
                        if (elem.getClassType().equals(Actor.class))
                            actorName = elem.getName();
                        else if (elem.getClassType().equals(UseCase.class))
                            ucName = elem.getName();
                    return new String[] {actorName, ucName};
                })
                .filter(el -> el[0].trim().length() > 0 && el[1].trim().length() > 0)
                .collect(Collectors.toSet());
    }

    public void extract(String prefix) {
        prefix = prefix == null ? "" : (prefix + "_");
        Set<UseCase> useCases = candidateElements.stream().filter(e -> e instanceof UseCase).map(e -> (UseCase)e).collect(Collectors.toSet());
        String output = String.join("\n", useCases.stream().map(NamedElement::getName).collect(Collectors.toSet()));
        writeFile(Paths.get("output", prefix + "useCases.csv").toString(), output);
        output = String.join("\n", extractAssociations(candidateElements).stream().map(p -> p[0] + ";" + p[1]).collect(Collectors.toSet()));
        writeFile(Paths.get("output", prefix + "actors_useCases.csv").toString(), output);
    }

    public void extractByDiagram(String prefix) {
        prefix = prefix == null ? "" : (prefix + "_");
        String filename = Paths.get("output", prefix + "actors_useCases_diag.csv").toString();
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8))) {
            for (Entry<DiagramPresentationElement, Collection<Element>> entry: elementsByDiagram.entrySet()) {
                String diagramName = entry.getKey().getName();
                if (diagramName == null) continue;
                Set<String[]> associations = extractAssociations(entry.getValue());
                String output = String.join("\n", associations.stream().map(p -> diagramName + ";" + p[0] + ";" + p[1]).collect(Collectors.toSet()));
                writer.write(output);
                writer.newLine();
            }
        } catch (IOException e) {
            Logger.getLogger(ElementNamesExtractor.class.getName()).log(Level.SEVERE, null, e);
        }
    }

    public void writeFile(String filename, String output) {
        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename), StandardCharsets.UTF_8))) {
            writer.write(output);
        } catch (IOException e) {
            Logger.getLogger(ElementNamesExtractor.class.getName()).log(Level.SEVERE, null, e);
        }
    }
}

